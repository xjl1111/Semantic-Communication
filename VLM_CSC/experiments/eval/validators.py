"""评估校验函数：SD 完整性检查、Fig8 检查点查找、变体标准化、协议校验。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

from eval.config import EvalConfig


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_sd_assets(sd_dir: Path) -> List[str]:
    required = [
        "model_index.json",
        "text_encoder",
        "unet",
        "vae",
        "scheduler",
        "tokenizer",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "tokenizer/tokenizer_config.json",
    ]
    missing = []
    for item in required:
        if not (sd_dir / item).exists():
            missing.append(item)
    return missing


def _get_fig8_variant_checkpoint(
    config: EvalConfig,
    variant: str,
    sender: str,
    stage_dataset: str,
) -> Path:
    if config.fig8_variant_checkpoint_map is None:
        raise RuntimeError("Fig8 strict mode requires fig8_variant_checkpoint_map")

    variant_map = config.fig8_variant_checkpoint_map.get(variant)
    if not isinstance(variant_map, dict):
        raise RuntimeError(f"Missing checkpoint variant map for {variant}")

    sender_map = variant_map.get(sender)
    if not isinstance(sender_map, dict):
        raise RuntimeError(f"Missing sender map for variant={variant}, sender={sender}")

    path = sender_map.get(stage_dataset)
    if not path:
        raise RuntimeError(
            f"Missing fig8 checkpoint for variant={variant}, sender={sender}, stage_dataset={stage_dataset}"
        )

    ckpt = Path(path).expanduser().resolve()
    if not ckpt.exists():
        raise RuntimeError(f"Fig8 task checkpoint missing: {ckpt}")
    return ckpt


def _normalize_fig8_med_variants(raw_variants) -> List[str]:
    if not isinstance(raw_variants, list) or len(raw_variants) == 0:
        raise RuntimeError("Fig8 strict protocol requires non-empty med_variants.")

    normalized: List[str] = []
    for item in raw_variants:
        if isinstance(item, bool):
            normalized.append("with_med" if item else "without_med")
            continue
        name = str(item).strip().lower()
        if name not in {"with_med", "without_med"}:
            raise RuntimeError(f"Unsupported fig8 med variant: {item}")
        normalized.append(name)

    if normalized != ["with_med", "without_med"]:
        raise RuntimeError("Fig8 strict protocol requires med_variants=['with_med','without_med']")
    return normalized


def _validate_fig8_strict_protocol_inputs(config: EvalConfig, metrics: List[str]) -> None:
    if config.channel_type != "rayleigh":
        raise RuntimeError("Fig8 strict protocol requires channel_type='rayleigh'")
    if metrics != ["bleu1", "bleu2"]:
        raise RuntimeError("Fig8 strict protocol requires metrics=['bleu1', 'bleu2']")
    if config.dataset_sequence != ["cifar", "birds", "catsvsdogs"]:
        raise RuntimeError("Fig8 strict protocol requires dataset_sequence=['cifar','birds','catsvsdogs']")
    _normalize_fig8_med_variants(config.med_variants)
    if config.eval_output_mode != "continual_learning_map":
        raise RuntimeError("Fig8 strict protocol requires eval_output_mode='continual_learning_map'")
    if config.dataset_roots is None and config.dataset_splits is None:
        raise RuntimeError("Fig8 strict protocol requires dataset_roots or dataset_splits")
    if str(config.test_split_dir).strip() != "":
        raise RuntimeError(
            "Fig8 strict protocol does not allow legacy test_split_dir. Use dataset_sequence + dataset_roots only."
        )
    if str(config.ckpt_blip).strip() != "" or str(config.ckpt_ram).strip() != "":
        raise RuntimeError(
            "Fig8 strict protocol uses checkpoint map only; legacy ckpt_blip/ckpt_ram must be empty."
        )
    if config.fig8_eval_snr_db is None:
        raise RuntimeError("Fig8 strict protocol requires fig8_eval_snr_db.")
    if isinstance(config.fig8_eval_snr_db, (list, tuple, dict)):
        raise RuntimeError("Fig8 strict protocol requires fig8_eval_snr_db to be a single numeric value.")
