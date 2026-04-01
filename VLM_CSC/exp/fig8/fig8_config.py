"""
Fig.8 配置 — Rayleigh 信道 MED 持续学习（BLEU-1/2 continual map）

╔══════════════════════════════════════════════════════════════════════════════╗
║  ★ 常用参数在下方「可调参数区」集中定义，直接修改数值即可；                ║
║    带 [论文] 标注的是论文明确给出的数值，修改后会偏离原论文设置。          ║
║    带 [⚠偏离] 标注的是当前值已与论文原始设置不同，需要知情后决策。        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import sys
from pathlib import Path

_EXP_DIR = Path(__file__).resolve().parent.parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from common.shared_config import SHARED_DEFAULTS, build_eval_block, build_shared_paths, build_train_block


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                        ║
# ║                    ★  训 练 参 数 区  ★                                ║
# ║                                                                        ║
# ║  以下参数仅影响训练过程（train_fig8.py），不影响评估。                  ║
# ║  修改后需要重新训练才能生效。                                          ║
# ║                                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ─── 信道 / 模型结构 ─────────────────────────────────────────────────────────

# 信道类型
#   [论文] Fig.8 使用 Rayleigh 衰落信道（§IV 实验设置），不建议修改
CHANNEL_TYPE: str = "rayleigh"   # ← 论文确定值

# 信道符号维度（= channel enc/dec 输出维度）
#   压缩率 ≈ max_text_len × bert_embed_dim / channel_dim
#   [论文] Tab.I 推测 channel_dim = 4，约 30:1 压缩
#   ▶ 与 Fig7 保持一致，确保论文对齐
CHANNEL_DIM: int = 4             # ← 论文推测值，与 Fig7 一致

# BLIP 侧最大 token 数
#   [论文] §III-A 明确写出：T = 24
#   但 24 token ≈ 15~18 英文单词，类别词（"bird"、"cat"）可能被截断
#   Fig8 跨 3 个不同数据集，描述长度差异更大，truncation 风险更高
#   ▶ 与 Fig7 保持一致，使用 30 减少截断风险
MAX_TEXT_LEN: int = 30           # ← 与 Fig7 一致，减少截断

# ─── 持续学习数据集顺序 ───────────────────────────────────────────────────────

# 数据集学习顺序（持续学习任务序列）
#   [论文] Fig.8 标注顺序为 CIFAR → Birds → CatsVsDogs
#   [⚠偏离] 修改顺序将直接改变 continual map 结果，不建议修改
DATASET_SEQUENCE: list = ["cifar", "birds", "catsvsdogs"]  # ← 论文确定值

# 是否启用 RAM sender（默认关闭：仅 BLIP）
ENABLE_RAM: bool = False

# ─── 训练 SNR 设置 ────────────────────────────────────────────────────────────

# 训练 SNR（固定点，与评估 SNR 一致）
#   [论文] 持续学习每个任务均在同一 SNR 下训练
TRAIN_SNR: float = 5.0            # ← 论文确定值

# ─── MED 超参数 ───────────────────────────────────────────────────────────────

# STM（短时记忆）最大样本数
#   [论文] §III-D 明确：N_STM 上限（论文给出数量级约 500）
MED_STM_MAX_SIZE: int = 500       # ← 论文参考值

# 知识蒸馏温度 τ
#   [论文] §III-D 公式中出现 τ，论文给出 τ = 10
MED_TAU: float = 10.0             # ← 论文确定值

# 迁移触发阈值（performance drop 超过此值才触发 MED 回放）
#   [论文] §III-D 给出 δ = 0.05
MED_THRESHOLD: float = 0.05       # ← 论文确定值

# ─── 优化器超参数 ─────────────────────────────────────────────────────────────

# 学习率
#   [论文] §IV-A 明确：lr = 1×10⁻⁴
TRAIN_LR: float = 1e-4            # ← 论文确定值

# 权重衰减
#   论文未指定；▶ 正向尝试：改为 1e-5 可提供轻微正则化
TRAIN_WD: float = 0.0

# mini-batch 大小
#   ▶ 从 8 提升到 16：梯度更稳定，MED 回放批次 = batch//2 = 8
#      总有效批次 16+8=24，接近 Fig7 的 32
TRAIN_BATCH_SIZE: int = 16

# ─── 训练阶段 epoch 数 ────────────────────────────────────────────────────────

# 论文未指定各阶段 epoch 数；以下为工程经验值
CHANNEL_EPOCHS:  int = 6          # Phase 1：信道重建阶段
SEMANTIC_EPOCHS: int = 8          # Phase 2：语义生成阶段
JOINT_EPOCHS:    int = 12         # Phase 3：联合阶段最大 epoch
#   ▶ 正向提升：JOINT_EPOCHS 可升至 20，结合 JOINT_PATIENCE 看收敛情况

# 早停 patience
#   ▶ 正向改进：改为 5 或 7，减少过早停止
JOINT_PATIENCE: int = 8

# ─── 联合阶段损失权重 ─────────────────────────────────────────────────────────

# [论文] §IV-B 明确：α = 0.5，β = 1.0
JOINT_ALPHA: float = 0.5          # ← 论文确定值
JOINT_BETA:  float = 1.0          # ← 论文确定值
JOINT_SCHEDULE: str = "alternate_steps"  # ← 论文交替优化协议，不建议修改


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                        ║
# ║                    ★  评 估 参 数 区  ★                                ║
# ║                                                                        ║
# ║  以下参数仅影响评估过程（eval_fig8.py），不影响训练。                   ║
# ║  修改后无需重新训练，直接重跑评估即可生效。                            ║
# ║                                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ─── 评估 SNR 设置 ────────────────────────────────────────────────────────────

# 评估固定 SNR（BLEU 在单一 SNR 下对比 with/without MED）
#   [论文] 论文 Fig.8 固定在 SNR=5dB 评估（正文 §IV-C）
EVAL_SNR_DB: float = 5.0          # ← 论文确定值

# ─── Stable Diffusion 生成参数 ────────────────────────────────────────────────

# DDIM 步数（Fig.8 以 BLEU 测文字质量，1 步足够做快速近似；提高可改善 BLEU）
#   ▶ 【快速验证】1 步足够；【最终出图】改为 10~20
SD_STEPS: int = 1

# SD 生成图像分辨率
#   ▶ 与 Fig7 保持一致，512×512 是论文图示分辨率
SD_HEIGHT: int = 512
SD_WIDTH:  int = 512

# ─── 评估采样控制 ─────────────────────────────────────────────────────────────

# 评估 mini-batch 大小
EVAL_BATCH_SIZE:    int = 8
# 最大 batch 数（-1 = 全部）
EVAL_MAX_BATCHES:   int = -1
# 每类最大样本数（-1 = 全部）
#   ▶ 【快速验证】改为 50；【最终出图】保持 -1
EVAL_MAX_PER_CLASS: int = -1

# ─── 图像描述模式 ─────────────────────────────────────────────────────────────
# BLIP 侧图像描述（caption）生成策略，直接影响 Level-A 文本准确率。
# 此变量同时控制训练（caption_cache 路径）和评估（checkpoint 路径）。
#
# 可选值：
#   "sr"        — SR（LANCZOS 上采样 32→256）+ BLIP-base + 简短 prompt，准确率 ~86%
#   "sr_prompt" — SR + BLIP-base + 增强 prompt（"a photo of an animal, a"），准确率 ~96%  ★ 推荐
#   "prompt"    — 无 SR + BLIP-base + 增强 prompt（对照 SR 贡献）
#   "blip2"     — SR + BLIP-2（~98%，需额外 ~5.5GB 显存 + 首次加载 ~25 分钟）
#
# ⚠  切换 CAPTION_MODE 后需要重新训练。
#    checkpoint 路径 = checkpoints/{CAPTION_MODE}/，模式不一致时 eval 会抛 RuntimeError。
CAPTION_MODE: str = "sr_prompt"   # sr / sr_prompt / prompt / blip2

# 是否使用在 CatsVsDogs 上微调过的 BLIP 模型（代替原始 BLIP-base）
#   True  — 使用 experiments/fig8/finetuned_blip/
#   False — 使用原始 BLIP-base
#
# ⚠  切换此项后需要重新生成 caption_cache 并重新训练。
USE_FINETUNED_BLIP: bool = False

CAPTION_PROMPT: str | None = None  # None = 使用 CAPTION_MODE 默认值


# ══════════════════════════════════════════════════════════════════════════════
#  配置构建函数（无需修改，自动读取上方两个参数区的常量）
# ══════════════════════════════════════════════════════════════════════════════

def build_fig8_config() -> dict:
    _root   = Path(__file__).resolve().parents[3]
    fig_dir = _root / "VLM_CSC" / "data" / "experiments" / "fig8"
    # USE_FINETUNED_BLIP=True 时加 _ft 后缀，避免微调/原始 BLIP 的检查点互相覆盖
    _ckpt_mode    = CAPTION_MODE + ("_ft" if USE_FINETUNED_BLIP else "")
    _cache_subdir = "caption_cache_ft" if USE_FINETUNED_BLIP else "caption_cache"

    cfg = {
        # ── 共享：路径 + 全局标量 ──────────────────────────────────────────────
        **build_shared_paths(),
        **SHARED_DEFAULTS,
    }

    if USE_FINETUNED_BLIP:
        _ft_blip_dir = fig_dir / "finetuned_blip"
        if not (_ft_blip_dir / "model.safetensors").exists():
            raise RuntimeError(
                f"USE_FINETUNED_BLIP=True 但微调模型不存在: {_ft_blip_dir}\n"
                f"请先把微调 BLIP 权重放到该目录，或将 USE_FINETUNED_BLIP 设为 False"
            )
        cfg["blip_ckb_dir"] = str(_ft_blip_dir)
        print(f"[fig8_config] 使用微调 BLIP: {_ft_blip_dir}")
    else:
        print(f"[fig8_config] 使用原始 BLIP-base: {cfg['blip_ckb_dir']}")

    cfg.update({
        # ── 协议锁 ────────────────────────────────────────────────────────────
        "protocol": {
            "name":                "fig8_rayleigh_continual_bleu_v1",
            "locked":              True,
            "required_metrics":    ["bleu1", "bleu2"],
            "require_strict_ckpt": True,
            "receiver_kb":         "sd",
        },

        # ── Fig.8 专有参数（读取顶部常量）────────────────────────────────────
        "senders":      ["blip", "ram"] if ENABLE_RAM else ["blip"],
        "channel_type": CHANNEL_TYPE,
        "channel_dim":  CHANNEL_DIM,
        "max_text_len": MAX_TEXT_LEN,
        "max_text_len_by_sender": {"blip": MAX_TEXT_LEN, "ram": 48} if ENABLE_RAM else {"blip": MAX_TEXT_LEN},
        "sd_height":    SD_HEIGHT,
        "sd_width":     SD_WIDTH,
        "dataset_sequence": DATASET_SEQUENCE,
        "dataset_roots": {
            "cifar":      str(_root / "data" / "datasets" / "cifar"),
            "birds":      str(_root / "data" / "datasets" / "birds"),
            "catsvsdogs": str(_root / "data" / "datasets" / "catsvsdogs"),
        },
        "dataset_splits": {
            "cifar": {
                "train": str(_root / "data" / "datasets" / "cifar"      / "train"),
                "val":   "",
                "test":  str(_root / "data" / "datasets" / "cifar"      / "test"),
            },
            "birds": {
                "train": str(_root / "data" / "datasets" / "birds"      / "train"),
                "val":   "",
                "test":  str(_root / "data" / "datasets" / "birds"      / "test"),
            },
            "catsvsdogs": {
                "train": str(_root / "data" / "datasets" / "catsvsdogs" / "train"),
                "val":   "",
                "test":  str(_root / "data" / "datasets" / "catsvsdogs" / "test"),
            },
        },
        "val_split_ratio":  0.2,
        "val_split_seed":   42,
        "med_variants":     ["with_med", "without_med"],
        "eval_output_mode": "continual_learning_map",
        "fig8_eval_snr_db": EVAL_SNR_DB,
        "comparison_axis":  "med",
        "snr_list":         [EVAL_SNR_DB],
        "sd_steps":         SD_STEPS,
        "metrics":          ["bleu1", "bleu2"],

        # ── 图像描述模式（SR 上采样默认开启，此键控制描述方法）─────────────────
        "caption_mode":       CAPTION_MODE,
        "caption_prompt":     CAPTION_PROMPT,
        "use_finetuned_blip": USE_FINETUNED_BLIP,
        "ckpt_mode":          _ckpt_mode,

        # ── Fig.8 实验路径 ─────────────────────────────────────────────────────
        "output_dir":               str(fig_dir),
        "train_monitor_output_dir": str(fig_dir / "train_monitor"),
        "final_eval_output_dir":    str(fig_dir / "final_eval"),
        "checkpoint_dir":           str(fig_dir / "checkpoints" / _ckpt_mode),
        "caption_cache_dir":        str(fig_dir / _cache_subdir),
        "finetuned_blip_dir":       str(fig_dir / "finetuned_blip"),

        # ── 训练块 ────────────────────────────────────────────────────────────
        "train": build_train_block(
            train_batch_size=TRAIN_BATCH_SIZE,
            train_snr_min_db=TRAIN_SNR,
            train_snr_max_db=TRAIN_SNR,
            train_snr_mode="fixed_point",
            channel_epochs=CHANNEL_EPOCHS,
            semantic_epochs=SEMANTIC_EPOCHS,
            joint_max_epochs=JOINT_EPOCHS,
            train_tag="fig8_train",
            use_previous_best_checkpoint=True,
            use_med=True,
            med_kwargs={
                "stm_max_size":       MED_STM_MAX_SIZE,
                "tau":                MED_TAU,
                "threshold":          MED_THRESHOLD,
                "transfer_if":        "greater",
                "strict_paper_repro": True,
            },
            train_lr=TRAIN_LR,
            train_wd=TRAIN_WD,
            joint_alpha=JOINT_ALPHA,
            joint_beta=JOINT_BETA,
            joint_patience=JOINT_PATIENCE,
            joint_schedule=JOINT_SCHEDULE,
        ),

        # ── 评估块 ────────────────────────────────────────────────────────────
        "eval": build_eval_block(
            tag="fig8_eval",
            batch_size=EVAL_BATCH_SIZE,
            max_batches=EVAL_MAX_BATCHES,
            max_per_class=EVAL_MAX_PER_CLASS,
            fig8_variant_checkpoint_map={
                "with_med": {
                    "blip": {
                        "cifar":      str(fig_dir / "checkpoints" / _ckpt_mode / "with_med" / "blip_1_cifar_phase_joint_best.pth"),
                        "birds":      str(fig_dir / "checkpoints" / _ckpt_mode / "with_med" / "blip_2_birds_phase_joint_best.pth"),
                        "catsvsdogs": str(fig_dir / "checkpoints" / _ckpt_mode / "with_med" / "blip_3_catsvsdogs_phase_joint_best.pth"),
                    },
                    **({
                        "ram": {
                            "cifar":      str(fig_dir / "checkpoints" / _ckpt_mode / "with_med" / "ram_1_cifar_phase_joint_best.pth"),
                            "birds":      str(fig_dir / "checkpoints" / _ckpt_mode / "with_med" / "ram_2_birds_phase_joint_best.pth"),
                            "catsvsdogs": str(fig_dir / "checkpoints" / _ckpt_mode / "with_med" / "ram_3_catsvsdogs_phase_joint_best.pth"),
                        },
                    } if ENABLE_RAM else {}),
                },
                "without_med": {
                    "blip": {
                        "cifar":      str(fig_dir / "checkpoints" / _ckpt_mode / "without_med" / "blip_1_cifar_phase_joint_best.pth"),
                        "birds":      str(fig_dir / "checkpoints" / _ckpt_mode / "without_med" / "blip_2_birds_phase_joint_best.pth"),
                        "catsvsdogs": str(fig_dir / "checkpoints" / _ckpt_mode / "without_med" / "blip_3_catsvsdogs_phase_joint_best.pth"),
                    },
                    **({
                        "ram": {
                            "cifar":      str(fig_dir / "checkpoints" / _ckpt_mode / "without_med" / "ram_1_cifar_phase_joint_best.pth"),
                            "birds":      str(fig_dir / "checkpoints" / _ckpt_mode / "without_med" / "ram_2_birds_phase_joint_best.pth"),
                            "catsvsdogs": str(fig_dir / "checkpoints" / _ckpt_mode / "without_med" / "ram_3_catsvsdogs_phase_joint_best.pth"),
                        },
                    } if ENABLE_RAM else {}),
                },
            },
        ),
    })

    return cfg
