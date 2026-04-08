"""
Fig.7 配置 — AWGN 信道 SSQ 曲线（BLIP [vs RAM] on CatsVsDogs）

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
# ║  以下参数仅影响训练过程（train_fig7.py），不影响评估。                  ║
# ║  修改后需要重新训练才能生效。                                          ║
# ║                                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ─── 信道 / 模型结构 ──────────────────────────────────────────────────────────

# 信道符号维度（= channel enc/dec 输出维度）
#   压缩率 ≈ max_text_len × bert_embed_dim / channel_dim
#   [⚠偏离] 论文 Tab.I 压缩率数字暗示 channel_dim 较小（推测原始值为 4）
#            channel_dim=4  → 约 32:1 压缩，接近论文设定，SSQ 会更低
#            channel_dim=128 → 约 1:1，几乎无瓶颈，SSQ 更高但偏离论文
#
#  「现有 checkpoint」: 32  — 修改此小节需要重新训练
#  「下次训练推荐」: 64  — 2:1 压缩，减少信道瓶颈对 SSQ 的损失
#
#  如需评估现有 checkpoint，请保持 CHANNEL_DIM=32。
CHANNEL_DIM: int = 4  # 30×4=120 符号，接近论文压缩率，信道成为瓶颈

# 是否启用 RAM sender 参与实验
#   True  — senders=["blip", "ram"]，训练和评估都包含 RAM
#   False — senders=["blip"]，仅 BLIP 参与训练和评估
ENABLE_RAM: bool = False

# BLIP 侧最大 token 数
#   [论文] §III-A 明确写出：T = 24
#
#  问题：24 token ≈ 15~18 英文单词，也就是 2~3 句话。
#  当 BLIP 生成较长描述时，类别词（"cat"/"dog"）可能被截断导致丢失。
#
#  「现有 checkpoint」: 24  — 修改此小节需要重新训练
#  「下次训练推荐」: 40  — 容刔1 句话 + 细节描述，降低截断风险
#
#  如需评估现有 checkpoint，请保持 MAX_TEXT_LEN=24。
MAX_TEXT_LEN: int = 30  # 改回 30

# ─── BLIP 描述生成模式 ────────────────────────────────────────────────────────

# BLIP 侧图像描述（caption）生成策略，直接影响 Level-A 文本准确率。
# 可选值：
#
#   "sr"        — SR（LANCZOS 上采样 32→256）+ BLIP-base + prompt="a photo of a"
#                  纯 SR 提升
#
#   "sr_prompt" — SR + BLIP-base + prompt="a photo of a"
#                  SR + 通用 prompt  ★ 推荐
#
#   "prompt"    — 无 SR + BLIP-base + prompt="a photo of a"
#                  纯 prompt（对照 SR 贡献）
#
#   "blip2"     — SR + BLIP-2 (blip2-opt-2.7b)
#                  准确率 ~98%，需要额外 ~5.5GB 显存 + 首次加载约 25 分钟
#
#  ⚠  切换 CAPTION_MODE 后需要重新训练，checkpoint 和 caption_cache 按模式自动隔离。
CAPTION_MODE: str = "sr_prompt"  # sr / sr_prompt / prompt / blip2

# ─── BLIP 模型选择 ────────────────────────────────────────────────────────────

# 是否使用在 CatsVsDogs 上微调过的 BLIP 模型（代替原始 BLIP-base）
#   True  — 使用 finetuned_blip/ 旧的 BLIP（A(src) ~86%）
#   False — 使用原始 BLIP-base（默认，实验公平性更强）
#
#  ⚠  切换此项后需要重新生成 caption_cache 并重新训练。
#        微调模型目录：  experiments/fig7/finetuned_blip/
#        运行微调： .venv\\Scripts\\python.exe VLM_CSC/experiments/fig7/finetune_blip_fig7.py
USE_FINETUNED_BLIP: bool = False  # True = 微调BLIP, False = 原始BLIP-base

# BLIP 描述生成提示词（可选，None = 按 CAPTION_MODE 默认值）
#
#   "a photo of a"            ← 所有模式默认（通用，无类别暗示）
#   "this is a"               ← 替代 prompt（可自由设置）
#   None                       ← BLIP-2 模式不使用 prompt
#
#  ⚠  修改 prompt 后需重新生成 caption_cache
CAPTION_PROMPT: str | None = ""   # "" = 真正无 prompt，BLIP 无条件生成（text=None）

# ─── CLIP 分类器微调 ──────────────────────────────────────────────────────────

# 是否在 CatsVsDogs 上微调 CLIP 分类器（用于 SSQ 评估）
#   True  — 训练一个 Linear(768→2) 分类头，评估时使用微调后的 CLIP
#   False — 使用 CLIP zero-shot（默认行为，与论文一致）
#
# ⚠  CLIP 微调仅影响评估指标的度量方式，不影响通信系统训练。
FINETUNE_CLIP: bool = True

# CLIP 微调超参数
CLIP_FT_EPOCHS: int = 10          # 微调 epoch 数
CLIP_FT_LR: float = 1e-3          # 分类头学习率（backbone 冻结，可用较大 lr）
CLIP_FT_BATCH_SIZE: int = 32      # 微调 batch 大小

# ─── 训练 SNR 设置 ────────────────────────────────────────────────────────────

# 训练时 SNR 均匀采样范围（uniform_range 模式）
#   论文：在测试 SNR 范围内均匀采样
#   [⚠偏离] 若将 SNR_LIST 上界改为 13，请同步将 TRAIN_SNR_MAX 改为 13.0
TRAIN_SNR_MIN: float = 0.0
TRAIN_SNR_MAX: float = 10.0     # 建议与 max(SNR_LIST) 保持一致

# ─── 优化器超参数 ─────────────────────────────────────────────────────────────

# 学习率
#   [论文] §IV-A 明确：lr = 1×10⁻⁴，不建议修改
TRAIN_LR: float = 1e-4         # ← 论文确定值

# 权重衰减（L2 正则化）
#   论文未指定；当前 0.0（无正则化）
#   ▶ 正向尝试：改为 1e-5 可提供轻微正则化，对 SSQ 影响通常 <0.3%
TRAIN_WD: float = 0.0

# mini-batch 大小
#   论文未明确；16 是折中值
#   ▶ 显存不足时改为 8；更大的 batch（32）通常轻微有益，需要更多显存
TRAIN_BATCH_SIZE: int = 32

# ─── 训练阶段 epoch 数 ────────────────────────────────────────────────────────

# 论文未指定各阶段 epoch 数；以下为工程经验值
CHANNEL_EPOCHS:  int = 5       # Phase 1：信道重建阶段  [快速验证: 20]
SEMANTIC_EPOCHS: int = 5       # Phase 2：语义生成阶段  [快速验证: 20]
JOINT_EPOCHS:    int = 5      # Phase 3：联合阶段最大 epoch  [快速验证: 20]
#   ▶ 正向提升：JOINT_EPOCHS 可升至 30~40，给联合微调更充分的时间

# 早停 patience（连续多少 epoch val_semantic_loss 无改善则停止）
#   论文未指定；设为 0 表示禁用早停，让联合训练跑完所有 epoch
#   正整数 N 表示连续 N 个 epoch 无改善则停止
JOINT_PATIENCE: int = 8  # 连续 N epoch val_semantic_loss 无改善则停止（0 = 禁用）

# ─── 联合阶段损失权重 ─────────────────────────────────────────────────────────

# L_total = JOINT_ALPHA × L_channel + JOINT_BETA × L_semantic
#   [论文] §IV-B 明确：α = 0.5，β = 1.0
JOINT_ALPHA: float = 0.5        # ← 论文确定值
JOINT_BETA:  float = 1.0        # ← 论文确定值

# 联合阶段交替优化策略
#   "alternate_steps"：每个 mini-batch 交替优化信道侧/语义侧（论文方式）
#   [⚠偏离] 不建议修改，否则偏离论文 Phase 3 训练协议
JOINT_SCHEDULE: str = "alternate_steps"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                        ║
# ║                    ★  评 估 参 数 区  ★                                ║
# ║                                                                        ║
# ║  以下参数仅影响评估过程（eval_fig7.py），不影响训练。                   ║
# ║  修改后无需重新训练，直接重跑评估即可生效。                            ║
# ║                                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ─── 图像描述模式 ─────────────────────────────────────────────────────────────
# CAPTION_MODE 已在「训练参数区」定义，评估时自动使用相同值加载对应 checkpoint。
#
#   可选值：
#     "sr"        — SR（LANCZOS 上采样 32→256）+ BLIP-base + 简短 prompt
#     "sr_prompt" — SR + BLIP-base + 增强 prompt，准确率 ~96%  ★ 推荐
#     "prompt"    — 无 SR + BLIP-base + 增强 prompt（对照 SR 贡献）
#     "blip2"     — SR + BLIP-2（~98%，需 ~5.5GB 额外显存）
#
# ⚠  评估所用 checkpoint 路径 = checkpoints/{CAPTION_MODE}/
#    如需切换模式，请在「训练参数区」修改 CAPTION_MODE 并重新训练。
#    模式不一致时 eval/core.py 会抛出 RuntimeError。

# ─── 评估 SNR 设置 ────────────────────────────────────────────────────────────

# 评估时的 SNR 点列表（图 x 轴，单位 dB）
#   [⚠偏离] 论文 Fig.7 x 轴目测约为 [1, 4, 7, 10, 13]（5 点）
#            当前 [0,2,4,6,8,10] 范围与论文不同，画图时 x 轴刻度不对应
#   ▶ 正向改进：改为 [1.0, 4.0, 7.0, 10.0, 13.0] 更贴近论文比较点
SNR_LIST: list = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

# ─── Stable Diffusion 生成参数 ────────────────────────────────────────────────

# DDIM 去噪步数（越多质量越高，但 ≥20 后收益递减）
#   论文未指定步数；≥20 均可，30 是合理默认值
#   ▶ 【快速验证】改为 10：速度提升 3×，CLIP 分类准确率几乎不变（SSQ 差 ≤1%）
#   ▶ 【最终出图】保持 30，确保图像质量
SD_STEPS:  int = 30             # ← 快速验证模式；最终出图改回 30

SD_HEIGHT: int = 512            # 已从 256 升级到 512，与论文图示分辨率一致 ✓
SD_WIDTH:  int = 512

# ─── 评估采样控制 ─────────────────────────────────────────────────────────────

# 评估 mini-batch 大小
EVAL_BATCH_SIZE:    int = 8
# 最大 batch 数（-1 = 全部）
EVAL_MAX_BATCHES:   int = -1    # 全量评估（消除抽样偏差）
# 每类最大样本数（-1 = 全部）
#   ▶ 【快速验证】改为 50，速度提升 5×，结果均值不变
#   ▶ 【最终出图】保持 -1 使用完整测试集
EVAL_MAX_PER_CLASS: int = 250     # 快速验证: 每类50张 × 2类 = 100张

# ══════════════════════════════════════════════════════════════════════════════
#  配置构建函数（无需修改，自动读取上方两个参数区的常量）
# ══════════════════════════════════════════════════════════════════════════════

def build_fig7_config() -> dict:
    _root   = Path(__file__).resolve().parents[3]
    fig_dir = _root / "VLM_CSC" / "data" / "experiments" / "fig7"

    # USE_FINETUNED_BLIP=True 时加 _ft 后缀，避免微调/原始 BLIP 的检查点互相覆盖
    _ckpt_mode    = CAPTION_MODE + ("_ft" if USE_FINETUNED_BLIP else "")
    _cache_subdir = "caption_cache_ft" if USE_FINETUNED_BLIP else "caption_cache"

    cfg = {
        # ── 共享：路径 + 全局标量 ──────────────────────────────────────────────
        **build_shared_paths(),
        **SHARED_DEFAULTS,
    }

    # 如果使用微调 BLIP，覆盖 blip_ckb_dir
    if USE_FINETUNED_BLIP:
        _ft_blip_dir = fig_dir / "finetuned_blip"
        if not (_ft_blip_dir / "model.safetensors").exists():
            raise RuntimeError(
                f"USE_FINETUNED_BLIP=True 但微调模型不存在: {_ft_blip_dir}\n"
                f"请先运行: .venv\\Scripts\\python.exe VLM_CSC/experiments/fig7/finetune_blip_fig7.py"
            )
        cfg["blip_ckb_dir"] = str(_ft_blip_dir)
        print(f"[fig7_config] 使用微调 BLIP: {_ft_blip_dir}")
    else:
        print(f"[fig7_config] 使用原始 BLIP-base: {cfg['blip_ckb_dir']}")

    cfg.update({
        # ── 协议锁 ────────────────────────────────────────────────────────────
        "protocol": {
            "name":                        "fig7_awgn_catsvsdogs_ssq_v1",
            "locked":                      True,
            "required_senders":            ["blip", "ram"] if ENABLE_RAM else ["blip"],
            "required_metrics":            ["ssq"],
            "required_classifier_backend": "clip_finetuned" if FINETUNE_CLIP else "clip_zeroshot",
            "require_strict_ckpt":         True,
            "receiver_kb":                 "sd",
        },

        # ── Fig.7 专有参数（读取顶部常量）────────────────────────────────────
        "senders":      ["blip", "ram"] if ENABLE_RAM else ["blip"],
        "channel_type": "awgn",
        "channel_dim":  CHANNEL_DIM,
        "max_text_len": MAX_TEXT_LEN,
        "caption_mode":       CAPTION_MODE,
        "caption_prompt":     CAPTION_PROMPT,
        "use_finetuned_blip": USE_FINETUNED_BLIP,
        "ckpt_mode":          _ckpt_mode,
        "max_text_len_by_sender": {"blip": MAX_TEXT_LEN, "ram": 48} if ENABLE_RAM else {"blip": MAX_TEXT_LEN},
        "snr_list":     SNR_LIST,
        "sd_steps":     SD_STEPS,
        "sd_height":    SD_HEIGHT,
        "sd_width":     SD_WIDTH,
        "metrics":      ["ssq"],

        # ── CLIP 分类器微调 ────────────────────────────────────────────────────
        "finetune_clip": FINETUNE_CLIP,
        "clip_ft_epochs": CLIP_FT_EPOCHS,
        "clip_ft_lr": CLIP_FT_LR,
        "clip_ft_batch_size": CLIP_FT_BATCH_SIZE,
        "clip_classifier_path": str(fig_dir / "finetuned_clip" / "clip_classifier.pth"),

        # ── Fig.7 库路径 ─────────────────────────────────────────────────────────────────
        "train_split_dir":   str(_root / "data" / "datasets" / "catsvsdogs" / "train"),
        "test_split_dir":    str(_root / "data" / "datasets" / "catsvsdogs" / "test"),
        "output_dir":        str(fig_dir),
        "checkpoint_dir":    str(fig_dir / "checkpoints" / _ckpt_mode),
        "caption_cache_dir": str(fig_dir / _cache_subdir),
        "finetuned_blip_dir": str(fig_dir / "finetuned_blip"),  # 微调模型目录（备用）

        # ── 训练块 ────────────────────────────────────────────────────────────
        "train": build_train_block(
            train_batch_size=TRAIN_BATCH_SIZE,
            train_snr_min_db=TRAIN_SNR_MIN,
            train_snr_max_db=TRAIN_SNR_MAX,
            train_snr_mode="uniform_range",
            channel_epochs=CHANNEL_EPOCHS,
            semantic_epochs=SEMANTIC_EPOCHS,
            joint_max_epochs=JOINT_EPOCHS,
            train_tag="fig7_train",
            use_previous_best_checkpoint=False,
            train_lr=TRAIN_LR,
            train_wd=TRAIN_WD,
            joint_alpha=JOINT_ALPHA,
            joint_beta=JOINT_BETA,
            joint_patience=JOINT_PATIENCE,
            joint_schedule=JOINT_SCHEDULE,
        ),

        # ── 评估块 ────────────────────────────────────────────────────────────
        # ▶ 快速验证：顶部 EVAL_MAX_PER_CLASS 改为 50，速度提升 5×
        # ▶ 三项叠加（SD_STEPS=10 + max_per_class=50 + 384分辨率）约 25× 加速
        "eval": build_eval_block(
            tag="fig7_eval",
            batch_size=EVAL_BATCH_SIZE,
            max_batches=EVAL_MAX_BATCHES,
            max_per_class=EVAL_MAX_PER_CLASS,
            ckpt_blip=str(fig_dir / "checkpoints" / _ckpt_mode / "blip_phase_joint_best.pth"),
            **({
                "ckpt_ram": str(fig_dir / "checkpoints" / _ckpt_mode / "ram_phase_joint_best.pth"),
            } if ENABLE_RAM else {}),
        ),
    })

    return cfg

