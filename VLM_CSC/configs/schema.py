"""Configuration schema for VLM-CSC experiments.

All fields are explicit so experiments can be reproduced by config snapshots.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BlipConfig:
    model_name: str = "Salesforce/blip-image-captioning-base"


@dataclass
class SdConfig:
    model_name: str = "sd-legacy/stable-diffusion-v1-5"


@dataclass
class RamConfig:
    checkpoint_path: str = ""


@dataclass
class DataConfig:
    image_size: int = 224
    max_caption_len: int = 32
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    datasets_root: str = "../data"


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 8
    semantic_layers: int = 3
    channel_hidden: List[int] = field(default_factory=lambda: [256, 128])
    nam_hidden: List[int] = field(default_factory=lambda: [56, 128, 56, 56])
    dropout: float = 0.1
    checkpoint_blip: str = "Salesforce/blip-image-captioning-base"
    checkpoint_sd: str = "sd-legacy/stable-diffusion-v1-5"


@dataclass
class ChannelConfig:
    channel_type: str = "awgn"
    snr_train_mode: str = "fixed"
    snr_train_db: float = 4.0
    snr_train_min_db: float = 0.0
    snr_train_max_db: float = 10.0
    seed: int = 42


@dataclass
class TrainConfig:
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 16
    grad_accum_steps: int = 1
    epochs: int = 10
    scheduler: str = "cosine"
    seeds: List[int] = field(default_factory=lambda: [1, 2, 3])
    med_enabled: bool = False
    nam_enabled: bool = True


@dataclass
class EvalConfig:
    snr_test_db: List[float] = field(default_factory=lambda: [float(v) for v in range(0, 11)])
    metrics: List[str] = field(default_factory=lambda: ["bleu", "ssq"])


@dataclass
class ExperimentConfig:
    exp_name: str = "default"
    figure: Optional[str] = None
    model_cache_dir: str = "D:/model_cache/vlm_csc"
    use_fp16: bool = True
    disable_fallback_in_formal_experiments: bool = True
    blip: BlipConfig = field(default_factory=BlipConfig)
    sd: SdConfig = field(default_factory=SdConfig)
    ram: RamConfig = field(default_factory=RamConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
