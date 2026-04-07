"""
Fig.9 配置 — AWGN 信道 NAM 消融实验（with/without NAM vs SNR，BLEU-1/2）

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
# ║  以下参数仅影响训练过程（train_fig9.py），不影响评估。                  ║
# ║  修改后需要重新训练才能生效。                                          ║
# ║                                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ─── 训练 SNR 设置 ────────────────────────────────────────────────────────────

# with_nam 实验的训练 SNR 范围（均匀采样）
#   [论文] with_nam 在全 SNR 范围训练以提升鲁棒性
NAM_TRAIN_SNR_MIN: float = 0.0    # ← 论文确定值
NAM_TRAIN_SNR_MAX: float = 10.0   # ← 论文确定值

# without_nam 消融实验的训练固定 SNR 点
#   [论文] Fig.9 对比列 = [0, 2, 4, 8] dB
NAM_ABLATION_SNRS: list = [0.0, 2.0, 4.0, 8.0]  # ← 论文确定值

# ─── 优化器超参数 ─────────────────────────────────────────────────────────────

# [论文] §IV-A 明确：lr = 1×10⁻⁴
TRAIN_LR: float = 1e-4            # ← 论文确定值
TRAIN_WD: float = 0.0             # 论文未指定
TRAIN_BATCH_SIZE: int = 8

# ─── 训练阶段 epoch 数 ────────────────────────────────────────────────────────

CHANNEL_EPOCHS:  int = 6          # Phase 1：信道重建阶段
SEMANTIC_EPOCHS: int = 6          # Phase 2：语义生成阶段
JOINT_EPOCHS:    int = 10         # Phase 3：联合阶段最大 epoch
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
# ║  以下参数仅影响评估过程（eval_fig9.py），不影响训练。                   ║
# ║  修改后无需重新训练，直接重跑评估即可生效。                            ║
# ║                                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ─── 评估 SNR 设置 ────────────────────────────────────────────────────────────

# 多密度测试 SNR 点列表（论文 Fig.9 x 轴）
#   [⚠偏离] 论文 Fig.9 x 轴目测约为 [1,2,...,10] 等间隔
#            当前 0~10 范围覆盖论文测试区域，可以保持
SNR_TEST_LIST: list = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# 稀疏评估点（用于 checkpoint 选择和每个实验组的评估）
SNR_LIST: list = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

# ─── Stable Diffusion 生成参数 ────────────────────────────────────────────────

# DDIM 步数（Fig.9 以 BLEU 测文字质量，1 步已足够做近似）
#   ▶ 【快速验证】1 步足够；【最终出图】改为 10~20
SD_STEPS: int = 1

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
#   True  — 使用 experiments/fig9/finetuned_blip/
#   False — 使用原始 BLIP-base
#
# ⚠  切换此项后需要重新生成 caption_cache 并重新训练。
USE_FINETUNED_BLIP: bool = False

CAPTION_PROMPT: str | None = None  # None = 使用 CAPTION_MODE 默认值


# ══════════════════════════════════════════════════════════════════════════════
#  配置构建函数（无需修改，自动读取上方两个参数区的常量）
# ══════════════════════════════════════════════════════════════════════════════

def build_fig9_config() -> dict:
    _root   = Path(__file__).resolve().parents[3]
    fig_dir = _root / "VLM_CSC" / "data" / "experiments" / "fig9"

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
        print(f"[fig9_config] 使用微调 BLIP: {_ft_blip_dir}")
    else:
        print(f"[fig9_config] 使用原始 BLIP-base: {cfg['blip_ckb_dir']}")

    cfg.update({
        # ── 协议锁 ────────────────────────────────────────────────────────────
        "protocol": {
            "name":                "fig9_awgn_nam_bleu_v1",
            "locked":              True,
            "required_metrics":    ["bleu1", "bleu2"],
            "require_strict_ckpt": True,
            "receiver_kb":         "sd",
        },

        # ── Fig.9 专有参数（读取顶部常量）────────────────────────────────────
        "senders":      ["blip"],
        "channel_type": "awgn",
        "snr_test_list": SNR_TEST_LIST,
        "nam_experiments": {
            "with_nam": {
                "use_nam":          True,
                "train_snr_mode":   "uniform_range",
                "train_snr_min_db": NAM_TRAIN_SNR_MIN,
                "train_snr_max_db": NAM_TRAIN_SNR_MAX,
            },
            **{
                f"without_nam_{int(snr)}": {
                    "use_nam":        False,
                    "train_snr_mode": "fixed_point",
                    "train_snr_db":   snr,
                }
                for snr in NAM_ABLATION_SNRS
            },
        },
        "snr_list": SNR_LIST,
        "sd_steps": SD_STEPS,
        "metrics":  ["bleu1", "bleu2"],

        # ── 图像描述模式（SR 上采样默认开启，此键控制描述方法）─────────────────
        "caption_mode":       CAPTION_MODE,
        "caption_prompt":     CAPTION_PROMPT,
        "use_finetuned_blip": USE_FINETUNED_BLIP,
        "ckpt_mode":          _ckpt_mode,

        # ── Fig.9 实验路径 ─────────────────────────────────────────────────────
        "train_split_dir":   str(_root / "data" / "datasets" / "catsvsdogs" / "train"),
        "test_split_dir":    str(_root / "data" / "datasets" / "catsvsdogs" / "test"),
        "output_dir":        str(fig_dir),
        "checkpoint_dir":    str(fig_dir / "checkpoints" / _ckpt_mode),
        "caption_cache_dir": str(fig_dir / _cache_subdir),
        "finetuned_blip_dir": str(fig_dir / "finetuned_blip"),

        # ── 训练块 ────────────────────────────────────────────────────────────
        "train": build_train_block(
            train_batch_size=TRAIN_BATCH_SIZE,
            train_snr_min_db=5.0,
            train_snr_max_db=5.0,
            train_snr_mode="uniform_range",
            channel_epochs=CHANNEL_EPOCHS,
            semantic_epochs=SEMANTIC_EPOCHS,
            joint_max_epochs=JOINT_EPOCHS,
            train_tag="fig9_train",
            use_previous_best_checkpoint=True,
            train_lr=TRAIN_LR,
            train_wd=TRAIN_WD,
            joint_alpha=JOINT_ALPHA,
            joint_beta=JOINT_BETA,
            joint_patience=JOINT_PATIENCE,
            joint_schedule=JOINT_SCHEDULE,
        ),

        # ── 评估块 ────────────────────────────────────────────────────────────
        "eval": build_eval_block(
            tag="fig9_eval",
            batch_size=EVAL_BATCH_SIZE,
            max_batches=EVAL_MAX_BATCHES,
            max_per_class=EVAL_MAX_PER_CLASS,
            nam_checkpoint_map={
                "with_nam": str(fig_dir / "checkpoints" / _ckpt_mode / "with_nam_phase_joint_best.pth"),
                **{
                    f"without_nam_{int(snr)}": str(
                        fig_dir / "checkpoints" / _ckpt_mode / f"without_nam_{int(snr)}_phase_joint_best.pth"
                    )
                    for snr in NAM_ABLATION_SNRS
                },
            },
        ),
    })

    return cfg
