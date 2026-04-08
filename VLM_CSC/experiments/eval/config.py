"""评估配置数据类。"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# Ensure VLM_CSC root is on sys.path so model imports work
_VLM_CSC_ROOT = Path(__file__).resolve().parents[2]
if str(_VLM_CSC_ROOT) not in sys.path:
    sys.path.insert(0, str(_VLM_CSC_ROOT))

# Import model package directly instead of loading from file
import model


@dataclass
class EvalConfig:
    project_root: str
    model_file: str
    target_file: str
    test_split_dir: str
    output_dir: str
    senders: List[str]
    blip_ckb_dir: str
    sd_ckb_dir: str
    channel_type: str
    ckpt_blip: str
    snr_list: List[float]
    metrics: List[str] | None = None
    ram_ckb_path: str = ""
    ckpt_ram: str = ""
    sd_steps: int = 20
    sd_height: int = 512
    sd_width: int = 512
    batch_size: int = 32
    max_batches: int = -1
    max_per_class: int = -1
    seed: int = 42
    strict_ckpt: bool = True
    strict_paper_repro: bool = True
    quiet_third_party: bool = True
    max_text_len: int = 24
    max_text_len_by_sender: Dict[str, int] | None = None
    tag: str = "eval"
    required_classifier_backend: str = "clip_zeroshot"
    baselines: List[str] | None = None
    dataset_sequence: List[str] | None = None
    continual_checkpoint_map: Dict[str, Dict[str, str]] | None = None
    baseline_checkpoints: Dict[str, str] | None = None
    export_alignment_examples: bool = False
    med_variants: List[bool] | None = None
    eval_output_mode: str | None = None
    fig_name: str = ""
    fig8_variant_checkpoint_map: Dict[str, Dict[str, Dict[str, str]]] | None = None
    dataset_roots: Dict[str, str] | None = None
    dataset_splits: Dict[str, Dict[str, str]] | None = None
    fig8_eval_snr_db: float | None = None
    training_snr_protocol: str = ""
    med_kwargs: Dict | None = None
    use_nam: bool = True
    channel_dim: int | None = None
    caption_mode: str = "baseline"
    caption_prompt: str | None = None
    clip_classifier_path: str = ""


def load_module(module_name: str = None, file_path: Path = None):
    """Return the model module. Arguments are kept for backward compatibility."""
    return model
