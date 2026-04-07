"""
所有实验（Fig.7~10）共享的配置帮助函数。

▶ 写在这里的（所有实验完全一致，不需要每个实验重复）：
  - 模型文件与预训练权重路径
  - 全局标量：seed, quiet_third_party, strict_ckpt
  - 训练超参数：optimizer lr=1e-4, weight_decay=0.0, val_ratio=0.2
              caption cache 开关, max_batches=-1
  - 每阶段默认值：channel_phase_objective, joint phase 的
              alpha/beta/schedule/patience/monitor
  - 评估默认值：batch_size=8, max_batches=-1, max_per_class=-1

▶ 写在各 figN_config.py 里的（实验之间有差异）：
  - protocol 块
  - channel_type, senders, metrics
  - snr_list, sd_steps, sd_height/sd_width, channel_dim
  - 每个阶段的 epoch 数
  - train_snr 范围与采样方式
  - train_batch_size（fig7=16，其他=8）
  - train_tag / eval tag
  - output_dir / checkpoint_dir / caption_cache_dir（按 fig 编号区分）
  - 实验特有字段（dataset_sequence, nam_experiments, baselines …）
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


# ── 路径 ──────────────────────────────────────────────────────────────────────

def build_shared_paths() -> Dict[str, str]:
    """返回 project_root 与所有预训练模型的绝对路径。

    shared_config.py 位于 VLM_CSC/exp/common/，向上三级即为 project_root。
    """
    project_root = Path(__file__).resolve().parents[3]
    exp_root     = project_root / "VLM_CSC" / "exp"
    model_dir    = project_root / "VLM_CSC" / "model"
    models_dir   = project_root / "VLM_CSC" / "data" / "assets" / "downloaded_models"
    return {
        "project_root": str(project_root),
        "model_file":   str(model_dir  / "VLM-CSC.py"),
        "target_file":  str(exp_root   / "eval" / "target.py"),
        "blip_ckb_dir": str(models_dir / "blip"),
        "ram_ckb_path": str(models_dir / "ram_swin_large_14m.pth"),
        "sd_ckb_dir":   str(models_dir / "sd15"),
    }


# ── 全局标量默认值 ────────────────────────────────────────────────────────────

SHARED_DEFAULTS: Dict[str, Any] = {
    "seed":             42,
    "quiet_third_party": True,
    "strict_ckpt":      True,
}


# ── 训练块 ────────────────────────────────────────────────────────────────────

def build_train_block(
    *,
    # ── 必须由各实验指定 ──
    train_batch_size:   int,
    train_snr_min_db:   float,
    train_snr_max_db:   float,
    train_snr_mode:     str,
    channel_epochs:     int,
    semantic_epochs:    int,
    joint_max_epochs:   int,
    train_tag:          str,
    # ── 可覆盖的开关 ──
    use_previous_best_checkpoint: bool = False,
    use_med:     bool = False,
    med_kwargs:  Optional[Dict[str, Any]] = None,
    # ── 可覆盖的超参数（各实验有个性需求时传入；否则使用共享默认值） ──
    train_lr:       float = 1e-4,      # 论文 §IV-A 明确：1×10⁻⁴
    train_wd:       float = 0.0,       # 论文未指定；默认 0.0
    joint_alpha:      float = 0.5,  # 论文 §IV-B 明确：α = 0.5
    joint_beta:       float = 1.0,  # 论文 §IV-B 明确：β = 1.0
    joint_patience:   int   = 8,    # 联合阶段早停 patience（0 = 禁用）
    channel_patience: int   = 8,    # 信道阶段早停 patience（0 = 禁用）
    semantic_patience: int  = 8,    # 语义阶段早停 patience（0 = 禁用）
    joint_schedule:   str   = "alternate_steps",  # 论文交替策略
) -> Dict[str, Any]:
    """构造完整的 train 配置块。

    共享默认值（在此固定，各实验不需要重复写）：
      lr=1e-4, weight_decay=0.0, val_ratio=0.2,
      use_caption_cache=True, strict_cache_required=True,
      channel_phase_objective="masked_sequence_mse",
      joint: alpha=0.5, beta=1.0, schedule="alternate_steps",
             early_stop_patience=3, monitor="val_semantic_loss"

    可覆盖参数（通过关键字参数传入 fig7_config.py 等）：
      train_lr, train_wd, joint_alpha, joint_beta,
      joint_patience, channel_patience, semantic_patience, joint_schedule

    必须由各实验指定的参数：
      train_batch_size, SNR 范围/模式, 各阶段 epoch 数, train_tag
    """
    block: Dict[str, Any] = {
        "enabled":                     True,
        "use_previous_best_checkpoint": use_previous_best_checkpoint,
        "train_lr":                    train_lr,
        "train_weight_decay":          train_wd,
        "train_batch_size":            train_batch_size,
        "train_max_batches":           -1,
        "train_max_per_class":         -1,
        "val_ratio":                   0.2,
        "train_snr_min_db":            train_snr_min_db,
        "train_snr_max_db":            train_snr_max_db,
        "train_snr_mode":              train_snr_mode,
        "use_caption_cache":           True,
        "strict_cache_required":       True,
        "train_phase_config": {
            "channel_phase": {
                "epochs":                   channel_epochs,
                "channel_phase_objective":  "masked_sequence_mse",
                "early_stop_patience":      channel_patience,
                "lr":                       train_lr,
                "weight_decay":             train_wd,
            },
            "semantic_phase": {
                "epochs":              semantic_epochs,
                "early_stop_patience": semantic_patience,
                "lr":                  train_lr,
                "weight_decay":        train_wd,
            },
            "joint_phase": {
                "max_joint_epochs":    joint_max_epochs,
                "early_stop_patience": joint_patience,
                "monitor":             "val_semantic_loss",
                "alpha":               joint_alpha,
                "beta":                joint_beta,
                "schedule":            joint_schedule,
                "lr":                  train_lr,
                "weight_decay":        train_wd,
            },
        },
        "train_tag": train_tag,
    }
    if use_med:
        block["use_med"]    = True
        block["med_kwargs"] = med_kwargs or {
            "stm_max_size":       500,
            "tau":                10.0,
            "threshold":          0.05,
            "transfer_if":        "greater",
            "strict_paper_repro": True,
        }
    return block


# ── 评估块 ────────────────────────────────────────────────────────────────────

def build_eval_block(
    *,
    tag: str,
    batch_size: int = 8,
    max_batches: int = -1,
    max_per_class: int = -1,
    **extra: Any,
) -> Dict[str, Any]:
    """构造完整的 eval 配置块。

    共享默认值：batch_size=8, max_batches=-1, max_per_class=-1。
    实验特有字段通过 **extra 传入（如 ckpt_blip=..., ckpt_ram=...）。
    """
    return {
        "enabled":       True,
        "batch_size":    batch_size,
        "max_batches":   max_batches,
        "max_per_class": max_per_class,
        "tag":           tag,
        **extra,
    }
