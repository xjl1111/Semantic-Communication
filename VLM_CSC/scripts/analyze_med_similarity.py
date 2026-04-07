"""
MED 相似度分布分析脚本

分析三个数据集样本间的 RBF 相似度分布，辅助选择合适的 STM→LTM 迁移阈值。

用法:
    python scripts/analyze_med_similarity.py
    python scripts/analyze_med_similarity.py --max_samples 200 --tau 10.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# 添加项目路径
_SCRIPT_DIR = Path(__file__).resolve().parent
_VLM_DIR = _SCRIPT_DIR.parent
_EXP_DIR = _VLM_DIR / "exp"
_PROJECT_ROOT = _VLM_DIR.parent

for _p in [str(_VLM_DIR), str(_EXP_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import TaskDatasetManager, build_vlm_system, load_module_from_file


def _print_ascii_histogram(data: np.ndarray, bins: int = 20, width: int = 50, current_threshold: float = 0.05):
    """在终端打印 ASCII 直方图。"""
    counts, edges = np.histogram(data, bins=bins)
    max_count = counts.max()

    print(f"  {'Range':^19} | {'Count':>6} | Distribution")
    print("  " + "-" * 19 + "-+-" + "-" * 6 + "-+-" + "-" * (width + 2))

    for i, count in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = "█" * bar_len

        # 标记当前阈值所在区间
        marker = ""
        if lo <= current_threshold < hi:
            marker = " ← threshold"

        print(f"  [{lo:.4f}, {hi:.4f}) | {count:6d} | {bar}{marker}")

    print()
    print(f"  当前阈值 {current_threshold} 位置: ", end="")
    if current_threshold < edges[0]:
        print(f"低于最小值 {edges[0]:.4f}")
    elif current_threshold >= edges[-1]:
        print(f"高于最大值 {edges[-1]:.4f}")
    else:
        pct_below = (data < current_threshold).mean() * 100
        print(f"{pct_below:.1f}% 的样本低于此阈值")


def compute_rbf_matrix(feats_a: torch.Tensor, feats_b: torch.Tensor, tau: float) -> torch.Tensor:
    """计算两组特征之间的 RBF 相似度矩阵。

    Args:
        feats_a: [N, D] 特征矩阵
        feats_b: [M, D] 特征矩阵
        tau: RBF 温度参数

    Returns:
        [N, M] 相似度矩阵
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    diff2 = torch.cdist(feats_a, feats_b, p=2).pow(2)
    return torch.exp(-diff2 / (2.0 * tau ** 2))


def extract_semantic_feature_from_text(model, text: str, device: str) -> torch.Tensor:
    """从文本提取 MED 使用的语义特征（masked mean pooling）。"""
    text_pack = model._encode_text([text], device=torch.device(device))
    src_ids = text_pack["token_ids"]
    src_mask = text_pack["attention_mask"]
    src_padding_mask = ~(src_mask.bool())

    src_embed = model._add_positional_encoding(model.embedding(src_ids))
    semantic_seq = model.semantic_encoder(src_embed, src_key_padding_mask=src_padding_mask, snr=0.0)

    # Masked mean pooling（与 MED 中计算方式一致）
    denom = src_mask.sum(dim=1, keepdim=True).clamp(min=1)
    masked_x = semantic_seq * src_mask.unsqueeze(-1)
    med_feature = masked_x.sum(dim=1) / denom  # [1, D]

    return med_feature[0]


def extract_features(
    model,
    records: list,
    device: str,
    max_samples: int,
    caption_cache: dict | None = None,
    dataset_name: str = "",
) -> tuple[torch.Tensor, list[str]]:
    """提取一组样本的语义特征。"""
    from PIL import Image

    features = []
    captions = []
    samples = records[:max_samples] if max_samples > 0 else records

    model.eval()
    with torch.no_grad():
        for rec in tqdm(samples, desc=f"Extracting [{dataset_name}]", leave=False):
            img_path = Path(rec["path"])
            img_key = str(img_path)

            # 获取 caption
            if caption_cache and img_key in caption_cache:
                caption = caption_cache[img_key]
            else:
                # 实时生成 caption
                pil_img = Image.open(img_path).convert("RGB")
                caption = model.sender_ckb.forward(pil_img)

            captions.append(caption)

            # 提取语义特征
            feat = extract_semantic_feature_from_text(model, caption, device)
            features.append(feat.cpu())

    return torch.stack(features, dim=0), captions  # [N, D], [N]


def analyze_similarity_distribution(
    model,
    task_manager: TaskDatasetManager,
    tau: float,
    max_samples: int,
    device: str,
    output_dir: Path,
):
    """分析三个数据集间的相似度分布。"""
    datasets = ["cifar", "birds", "catsvsdogs"]
    dataset_features = {}

    print("\n" + "=" * 60)
    print("  Step 1: 提取各数据集的语义特征")
    print("=" * 60)

    for ds_name in datasets:
        print(f"\n[{ds_name}] 提取特征...")
        records = task_manager.get_task_train_set(ds_name)
        feats, captions = extract_features(model, records, device, max_samples, dataset_name=ds_name)
        dataset_features[ds_name] = feats
        print(f"  -> 特征形状: {feats.shape}, 范数均值: {feats.norm(dim=1).mean():.4f}")
        print(f"  -> 示例 caption: {captions[0][:80]}...")

    print("\n" + "=" * 60)
    print("  Step 2: 计算相似度分布")
    print("=" * 60)

    # 存储所有相似度
    intra_sims = {}  # 同数据集内
    inter_sims = {}  # 不同数据集间
    all_sims = []

    for i, ds_i in enumerate(datasets):
        feats_i = dataset_features[ds_i]

        # 同数据集内相似度（上三角，排除对角线）
        sim_matrix = compute_rbf_matrix(feats_i, feats_i, tau)
        n = sim_matrix.shape[0]
        triu_indices = torch.triu_indices(n, n, offset=1)
        intra = sim_matrix[triu_indices[0], triu_indices[1]].numpy()
        intra_sims[ds_i] = intra
        all_sims.extend(intra.tolist())

        # 与其他数据集的相似度
        for j, ds_j in enumerate(datasets):
            if j <= i:
                continue
            feats_j = dataset_features[ds_j]
            sim_matrix = compute_rbf_matrix(feats_i, feats_j, tau)
            inter = sim_matrix.flatten().numpy()
            inter_sims[f"{ds_i}_vs_{ds_j}"] = inter
            all_sims.extend(inter.tolist())

    all_sims = np.array(all_sims)

    # 打印统计信息
    print(f"\n[全局统计] tau={tau}")
    print(f"  相似度范围: [{all_sims.min():.6f}, {all_sims.max():.6f}]")
    print(f"  均值: {all_sims.mean():.6f}, 标准差: {all_sims.std():.6f}")
    print(f"  分位数: 5%={np.percentile(all_sims, 5):.6f}, "
          f"25%={np.percentile(all_sims, 25):.6f}, "
          f"50%={np.percentile(all_sims, 50):.6f}, "
          f"75%={np.percentile(all_sims, 75):.6f}, "
          f"95%={np.percentile(all_sims, 95):.6f}")

    print("\n[同数据集内相似度]")
    for ds_name, sims in intra_sims.items():
        print(f"  {ds_name}: min={sims.min():.6f}, mean={sims.mean():.6f}, "
              f"max={sims.max():.6f}, std={sims.std():.6f}")

    print("\n[跨数据集相似度]")
    for pair_name, sims in inter_sims.items():
        print(f"  {pair_name}: min={sims.min():.6f}, mean={sims.mean():.6f}, "
              f"max={sims.max():.6f}, std={sims.std():.6f}")

    # ASCII 直方图
    print("\n[相似度分布直方图]")
    _print_ascii_histogram(all_sims, bins=20, width=50, current_threshold=0.05)

    # 绘制分布图
    print("\n" + "=" * 60)
    print("  Step 3: 生成可视化图表")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 图1: 整体相似度分布直方图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 全局分布
    ax = axes[0, 0]
    ax.hist(all_sims, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(0.05, color='red', linestyle='--', linewidth=2, label=f'Current threshold (0.05)')
    ax.axvline(np.percentile(all_sims, 30), color='orange', linestyle='--', linewidth=2,
               label=f'30th percentile ({np.percentile(all_sims, 30):.4f})')
    ax.axvline(np.percentile(all_sims, 50), color='green', linestyle='--', linewidth=2,
               label=f'50th percentile ({np.percentile(all_sims, 50):.4f})')
    ax.set_xlabel('RBF Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Overall RBF Similarity Distribution (tau={tau})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 同数据集内分布对比
    ax = axes[0, 1]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for (ds_name, sims), color in zip(intra_sims.items(), colors):
        ax.hist(sims, bins=50, density=True, alpha=0.5, label=f'{ds_name}', color=color)
    ax.set_xlabel('RBF Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Intra-Dataset Similarity Distributions', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 跨数据集分布对比
    ax = axes[1, 0]
    colors = ['#d62728', '#9467bd', '#8c564b']
    for (pair_name, sims), color in zip(inter_sims.items(), colors):
        ax.hist(sims, bins=50, density=True, alpha=0.5, label=pair_name, color=color)
    ax.set_xlabel('RBF Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Inter-Dataset Similarity Distributions', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 阈值选择辅助图
    ax = axes[1, 1]
    percentiles = np.arange(5, 100, 5)
    thresholds = [np.percentile(all_sims, p) for p in percentiles]
    ax.plot(percentiles, thresholds, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax.axhline(0.05, color='red', linestyle='--', linewidth=2, label='Current threshold (0.05)')
    ax.set_xlabel('Percentile (%)', fontsize=12)
    ax.set_ylabel('Similarity Threshold', fontsize=12)
    ax.set_title('Threshold Selection Guide\n(transfer_if="smaller": samples below threshold go to LTM)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 添加百分比注释
    for p, t in zip(percentiles[::2], thresholds[::2]):
        ax.annotate(f'{t:.3f}', (p, t), textcoords="offset points", xytext=(0, 8),
                   ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    out_png = output_dir / f"med_similarity_distribution_tau{tau}.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> 分布图已保存: {out_png}")

    # 图2: 详细分位数图
    fig, ax = plt.subplots(figsize=(12, 6))

    # 准备数据
    data_labels = list(intra_sims.keys()) + list(inter_sims.keys())
    data_values = list(intra_sims.values()) + list(inter_sims.values())

    positions = np.arange(len(data_labels))
    bp = ax.boxplot(data_values, positions=positions, widths=0.6, patch_artist=True)

    # 设置颜色
    colors_box = ['#1f77b4', '#ff7f0e', '#2ca02c'] + ['#d62728', '#9467bd', '#8c564b']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(0.05, color='red', linestyle='--', linewidth=2, label='Current threshold (0.05)')
    ax.set_xticks(positions)
    ax.set_xticklabels(data_labels, rotation=30, ha='right')
    ax.set_ylabel('RBF Similarity', fontsize=12)
    ax.set_title(f'Similarity Distribution by Dataset Pair (tau={tau})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_png2 = output_dir / f"med_similarity_boxplot_tau{tau}.png"
    plt.savefig(out_png2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> 箱线图已保存: {out_png2}")

    # 输出阈值建议
    print("\n" + "=" * 60)
    print("  Step 4: 阈值选择建议")
    print("=" * 60)

    print(f"\n当前配置: threshold={0.05}, transfer_if='smaller'")
    print(f"含义: 与 LTM 平均相似度 < 0.05 的样本（即最不相似的样本）才会进入 LTM")

    pct_below_current = (all_sims < 0.05).mean() * 100
    print(f"\n当前阈值下: {pct_below_current:.1f}% 的样本对相似度 < 0.05")

    print("\n建议阈值（基于分位数）:")
    for pct in [10, 20, 30, 40, 50]:
        threshold = np.percentile(all_sims, pct)
        print(f"  {pct}th percentile -> threshold = {threshold:.6f}")

    print("\n如果希望约 30-50% 的 STM 样本能进入 LTM:")
    print(f"  建议 threshold 设为 {np.percentile(all_sims, 40):.4f} ~ {np.percentile(all_sims, 60):.4f}")

    # 保存详细统计到文件
    stats_file = output_dir / f"med_similarity_stats_tau{tau}.txt"
    with stats_file.open("w", encoding="utf-8") as f:
        f.write(f"MED Similarity Analysis (tau={tau})\n")
        f.write("=" * 60 + "\n\n")

        f.write("[Global Statistics]\n")
        f.write(f"  Range: [{all_sims.min():.6f}, {all_sims.max():.6f}]\n")
        f.write(f"  Mean: {all_sims.mean():.6f}, Std: {all_sims.std():.6f}\n\n")

        f.write("[Percentiles]\n")
        for pct in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]:
            f.write(f"  {pct:2d}%: {np.percentile(all_sims, pct):.6f}\n")

        f.write("\n[Intra-Dataset Statistics]\n")
        for ds_name, sims in intra_sims.items():
            f.write(f"  {ds_name}: min={sims.min():.6f}, mean={sims.mean():.6f}, "
                   f"max={sims.max():.6f}, std={sims.std():.6f}\n")

        f.write("\n[Inter-Dataset Statistics]\n")
        for pair_name, sims in inter_sims.items():
            f.write(f"  {pair_name}: min={sims.min():.6f}, mean={sims.mean():.6f}, "
                   f"max={sims.max():.6f}, std={sims.std():.6f}\n")

    print(f"\n  -> 统计报告已保存: {stats_file}")

    # 额外分析：不同 tau 值的影响
    print("\n" + "=" * 60)
    print("  Step 5: 不同 tau 值对相似度分布的影响")
    print("=" * 60)

    # 计算原始欧氏距离
    all_feats = torch.cat([dataset_features[ds] for ds in datasets], dim=0)
    dist_matrix = torch.cdist(all_feats, all_feats, p=2)
    n_feat = dist_matrix.shape[0]
    triu_indices = torch.triu_indices(n_feat, n_feat, offset=1)
    all_distances = dist_matrix[triu_indices[0], triu_indices[1]].numpy()

    print(f"\n[欧氏距离统计]")
    print(f"  范围: [{all_distances.min():.4f}, {all_distances.max():.4f}]")
    print(f"  均值: {all_distances.mean():.4f}, 标准差: {all_distances.std():.4f}")

    tau_values = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0]
    print(f"\n[不同 tau 下的相似度分布]")
    tau_stats = []
    for test_tau in tau_values:
        sim_values = np.exp(-all_distances**2 / (2.0 * test_tau**2))
        stats = {
            "tau": test_tau,
            "min": sim_values.min(),
            "mean": sim_values.mean(),
            "max": sim_values.max(),
            "p10": np.percentile(sim_values, 10),
            "p30": np.percentile(sim_values, 30),
            "p50": np.percentile(sim_values, 50),
        }
        tau_stats.append(stats)
        print(f"  tau={test_tau:5.1f}: min={stats['min']:.4f}, mean={stats['mean']:.4f}, "
              f"max={stats['max']:.4f}, p30={stats['p30']:.4f}, p50={stats['p50']:.4f}")

    # 绘制 tau 敏感性图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for test_tau in [5.0, 10.0, 20.0, 50.0]:
        sim_vals = np.exp(-all_distances**2 / (2.0 * test_tau**2))
        ax.hist(sim_vals, bins=50, density=True, alpha=0.5, label=f'tau={test_tau}')
    ax.set_xlabel('RBF Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Effect of tau on Similarity Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    taus = [s["tau"] for s in tau_stats]
    ax.plot(taus, [s["p30"] for s in tau_stats], 'o-', label='30th percentile', linewidth=2)
    ax.plot(taus, [s["p50"] for s in tau_stats], 's-', label='50th percentile', linewidth=2)
    ax.plot(taus, [s["mean"] for s in tau_stats], '^-', label='Mean', linewidth=2)
    ax.axhline(0.05, color='red', linestyle='--', label='Current threshold (0.05)')
    ax.set_xlabel('tau', fontsize=12)
    ax.set_ylabel('Similarity', fontsize=12)
    ax.set_title('Similarity Percentiles vs tau', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    tau_png = output_dir / f"med_tau_sensitivity.png"
    plt.savefig(tau_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  -> tau 敏感性图已保存: {tau_png}")

    return {
        "all_sims": all_sims,
        "intra_sims": intra_sims,
        "inter_sims": inter_sims,
        "all_distances": all_distances,
        "tau_stats": tau_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="MED 相似度分布分析")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="每个数据集最大采样数（默认 100）")
    parser.add_argument("--tau", type=float, default=10.0,
                        help="RBF 温度参数（默认 10.0，与 MED 配置一致）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备（cuda/cpu）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录")
    args = parser.parse_args()

    device = args.device
    tau = args.tau
    max_samples = args.max_samples
    output_dir = Path(args.output_dir) if args.output_dir else _VLM_DIR / "data" / "experiments" / "fig8" / "similarity_analysis"

    print("=" * 60)
    print("  MED 相似度分布分析")
    print("=" * 60)
    print(f"  tau: {tau}")
    print(f"  max_samples: {max_samples}/dataset")
    print(f"  device: {device}")
    print(f"  output_dir: {output_dir}")

    # 加载模型
    print("\n[Loading] 加载 VLM 模型...")
    model_file = _VLM_DIR / "model" / "VLM-CSC.py"
    vlm_module = load_module_from_file("VLM_CSC_model", model_file)

    # 使用 fig8 配置
    sys.path.insert(0, str(_EXP_DIR / "fig8"))
    from fig8_config import build_fig8_config
    cfg = build_fig8_config()

    model = build_vlm_system(
        vlm_module,
        sender="blip",
        blip_dir=Path(cfg["blip_ckb_dir"]),
        ram_ckpt=Path(cfg["ram_ckb_path"]) if cfg.get("ram_ckb_path") else None,
        sd_dir=Path(cfg["sd_ckb_dir"]),
        channel_type=cfg["channel_type"],
        device=device,
        quiet_third_party=True,
        use_real_receiver_ckb=False,  # 不需要加载 SD
        enable_med=False,  # 分析时不需要 MED
        med_kwargs=None,
        max_text_len=cfg["max_text_len"],
        caption_mode=cfg.get("caption_mode", "sr_prompt"),
        caption_prompt=cfg.get("caption_prompt"),
    )

    # 创建数据集管理器
    task_manager = TaskDatasetManager(
        sequence=cfg["dataset_sequence"],
        dataset_roots=cfg["dataset_roots"],
        dataset_splits=cfg["dataset_splits"],
        max_per_class=max_samples,
        val_split_ratio=cfg["val_split_ratio"],
        val_split_seed=cfg["val_split_seed"],
        strict_mode=False,
        consumer="audit",
    )

    # 运行分析
    results = analyze_similarity_distribution(
        model=model,
        task_manager=task_manager,
        tau=tau,
        max_samples=max_samples,
        device=device,
        output_dir=output_dir,
    )

    print("\n" + "=" * 60)
    print("  分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
