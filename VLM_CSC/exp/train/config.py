"""训练配置数据类、种子设置、模块加载与阶段配置验证。"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from common.utils import load_module_from_file
from train.phase_utils import require_phase_block as _require_phase_block


@dataclass
class TrainConfig:
    project_root: str
    model_file: str
    train_split_dir: str
    output_dir: str
    checkpoint_dir: str
    senders: List[str]
    blip_ckb_dir: str
    sd_ckb_dir: str
    channel_type: str = "awgn"
    ram_ckb_path: str = ""
    seed: int = 42
    quiet_third_party: bool = True
    strict_paper_repro: bool = True
    max_text_len: int = 24
    max_text_len_by_sender: Dict[str, int] | None = None
    train_epochs: int = 1
    train_lr: float = 1e-4
    train_weight_decay: float = 0.0
    train_batch_size: int = 8
    train_max_batches: int = -1
    train_max_per_class: int = -1
    val_ratio: float = 0.2
    train_snr_min_db: float = 5.0
    train_snr_max_db: float = 5.0
    train_snr_mode: str = "uniform_range"
    use_caption_cache: bool = True
    caption_cache_dir: str = ""
    strict_cache_required: bool = True
    train_phase_config: Dict[str, Dict] | None = None
    train_tag: str = "train"
    fig_name: str = ""
    dataset_sequence: List[str] | None = None
    dataset_roots: Dict[str, str] | None = None
    dataset_splits: Dict[str, Dict[str, str]] | None = None
    val_split_ratio: float = 0.2
    val_split_seed: int = 42
    med_kwargs: Dict | None = None
    auto_rebuild_cache_on_hash_mismatch: bool = True
    use_nam: bool = True
    channel_dim: int | None = None
    caption_mode: str = "baseline"
    caption_prompt: str | None = None
    sd_steps: int = 20


def set_seed(seed: int = 42) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_vlm_module(vlm_file: Path):
    return load_module_from_file("vlm_csc_module", vlm_file)


def _validate_train_phase_config(config: TrainConfig) -> Dict[str, Dict]:
    if config.train_phase_config is None:
        raise RuntimeError("train_phase_config is required under strict protocol.")
    phase_cfg = config.train_phase_config
    channel = _require_phase_block(phase_cfg, "channel_phase", ["epochs", "lr", "weight_decay"])

    if "channel_phase_objective" not in channel:
        if config.strict_paper_repro:
            raise RuntimeError("Strict mode requires channel_phase.channel_phase_objective to be explicitly configured.")
        legacy_obj = channel.get("objective")
        if legacy_obj is None:
            raise RuntimeError("channel_phase_objective is missing and no legacy objective is provided.")
        legacy_map = {
            "sequence_mse_proxy": "masked_sequence_mse",
            "contrastive_info_nce": "info_nce_sequence",
        }
        mapped = legacy_map.get(str(legacy_obj).lower())
        if mapped is None:
            raise RuntimeError(f"Unsupported legacy channel objective: {legacy_obj}")
        channel["channel_phase_objective"] = mapped

    channel_objective = str(channel["channel_phase_objective"]).lower()
    if channel_objective not in {"masked_sequence_mse", "info_nce_sequence"}:
        raise RuntimeError(
            f"channel_phase_objective must be one of ['masked_sequence_mse', 'info_nce_sequence'], got {channel_objective}"
        )
    if channel_objective == "info_nce_sequence":
        if "info_nce_temperature" in channel and float(channel["info_nce_temperature"]) <= 0:
            raise RuntimeError("channel_phase.info_nce_temperature must be > 0 when using info_nce_sequence")

    semantic = _require_phase_block(phase_cfg, "semantic_phase", ["epochs", "lr", "weight_decay"])
    joint = _require_phase_block(
        phase_cfg,
        "joint_phase",
        ["max_joint_epochs", "early_stop_patience", "monitor", "alpha", "beta", "schedule", "lr", "weight_decay"],
    )
    if int(channel["epochs"]) <= 0 or int(semantic["epochs"]) <= 0:
        raise RuntimeError("Phase epochs must be > 0")
    if int(joint["max_joint_epochs"]) <= 0:
        raise RuntimeError("joint_phase.max_joint_epochs must be > 0")
    if float(joint["alpha"]) <= 0 or float(joint["beta"]) <= 0:
        raise RuntimeError("joint_phase alpha and beta must be positive")
    return phase_cfg
