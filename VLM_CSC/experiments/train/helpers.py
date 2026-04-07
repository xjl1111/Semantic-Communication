"""训练内部辅助函数：SNR 采样、checkpoint 操作、MED 批次、BLEU、CSV 等。"""
from __future__ import annotations

import csv
import importlib
import random
import re
from pathlib import Path
from typing import Dict, List

from PIL import Image

from common.utils import chunk_records
from train.phase_utils import (
    assert_channel_forward_contract as _assert_channel_forward_contract,
    assert_semantic_forward_contract as _assert_semantic_forward_contract,
    masked_sequence_mse as _masked_sequence_mse,
    compute_info_nce_sequence as _compute_info_nce_sequence,
)


# ─── 训练期间 Level-A/B 实时监控用 cat/dog 词汇正则 ───────────────────────
_TRAIN_CAT_PAT = re.compile(r"\b(cat|cats|kitten|kittens|kitty|feline)\b", re.IGNORECASE)
_TRAIN_DOG_PAT = re.compile(r"\b(dog|dogs|puppy|puppies|pup|canine|hound)\b", re.IGNORECASE)


def _train_text_matches_label(text: str, label: int) -> bool:
    """训练监控辅助函数：文本是否包含与指定整数标签对应的类别词 (0=cat, 1=dog)。"""
    if label == 0:
        return bool(_TRAIN_CAT_PAT.search(text))
    if label == 1:
        return bool(_TRAIN_DOG_PAT.search(text))
    return False


def _snr_sampler(min_db: float, max_db: float, rng: random.Random) -> float:
    if min_db > max_db:
        raise RuntimeError(f"train_snr_min_db must be <= train_snr_max_db, got {min_db} > {max_db}")
    if abs(max_db - min_db) <= 1e-12:
        return float(min_db)
    return float(rng.uniform(min_db, max_db))


def _resolve_train_snr(
    snr_mode: str,
    min_db: float,
    max_db: float,
    rng: random.Random,
) -> float:
    if snr_mode == "fixed_point":
        if abs(max_db - min_db) > 1e-12:
            raise RuntimeError(
                f"snr_train_mode=fixed_point requires train_snr_min_db == train_snr_max_db, got {min_db} vs {max_db}"
            )
        return float(min_db)

    if snr_mode == "uniform_range":
        return _snr_sampler(min_db=min_db, max_db=max_db, rng=rng)

    raise RuntimeError(f"Unknown snr_train_mode: {snr_mode}. Expected one of: fixed_point, uniform_range")


def _load_phase_best_checkpoint_strict(*, model, checkpoint_path: Path, phase_name: str) -> None:
    import torch

    if not checkpoint_path.exists():
        raise RuntimeError(
            f"{phase_name} strict handoff requires best checkpoint to exist, but not found: {checkpoint_path}"
        )
    try:
        state = torch.load(checkpoint_path, map_location=next(model.parameters()).device)
        if not isinstance(state, dict) or "state_dict" not in state:
            raise RuntimeError(
                f"{phase_name} best checkpoint format invalid: missing 'state_dict' in {checkpoint_path}"
            )
        model.load_state_dict(state["state_dict"], strict=True)
    except Exception as exc:
        raise RuntimeError(
            f"{phase_name} strict handoff failed while loading best checkpoint: {checkpoint_path}; error={exc}"
        ) from exc


def _sample_memory_batch_for_step(
    *,
    model,
    med_replay_enabled: bool,
    replay_batch_size: int,
    stm_ratio: float,
) -> List[Dict[str, object]]:
    if not med_replay_enabled:
        if model.med is not None:
            raise RuntimeError("without_med strict mode requires model.med to be None.")
        return []

    replay_batch_size = int(replay_batch_size)
    if replay_batch_size <= 0:
        return []

    replay_samples = model.sample_med_batch(batch_size=replay_batch_size, stm_ratio=float(stm_ratio))
    return [
        {
            "source_text": str(s.caption_text),
            "image_id": str(s.image_id),
            "dataset_id": str(s.dataset_id),
            "label": None,
            "is_memory_sample": True,
        }
        for s in replay_samples
    ]


def _prepare_merged_batch(
    *,
    batch,
    caption_cache: Dict[str, str],
    dataset_id: str,
    model,
    med_replay_enabled: bool,
    med_replay_batch_size: int,
    med_replay_stm_ratio: float,
    phase_label: str,
) -> tuple:
    """Build current samples, optionally sample MED memory, merge, and validate.

    Returns ``(merged_batch, current_batch, has_memory)``.
    """
    current_batch: List[Dict[str, object]] = []
    for rec in batch:
        text = caption_cache.get(str(rec["path"]))
        if not text or not str(text).strip():
            raise RuntimeError(f"Missing source_text in caption cache for image={rec['path']}")
        current_batch.append({
            "source_text": str(text),
            "image_id": str(rec["path"].name),
            "dataset_id": str(dataset_id),
            "label": rec.get("label"),
            "is_memory_sample": False,
        })

    memory_batch = _sample_memory_batch_for_step(
        model=model,
        med_replay_enabled=med_replay_enabled,
        replay_batch_size=med_replay_batch_size,
        stm_ratio=med_replay_stm_ratio,
    )
    merged = list(current_batch) + list(memory_batch)
    # integrity assertion
    if not med_replay_enabled and len(memory_batch) > 0:
        raise RuntimeError(f"without_med {phase_label} cannot have memory samples")
    return merged, current_batch, len(memory_batch) > 0


def _log_med_state(model, phase: str, epoch: int, step: int = -1) -> None:
    """Log MED STM/LTM sizes for monitoring memory evolution."""
    if model.med is None:
        return
    med = model.med
    step_str = f" step={step}" if step >= 0 else ""
    transfer_info = ""
    if med.is_stm_full():
        transfer_info = " [STM FULL - transfer pending]"
    print(
        f"[MED][{phase}] epoch={epoch}{step_str}  "
        f"STM={len(med.stm)}/{med.stm_max_size}  "
        f"LTM={len(med.ltm)}"
        f"{transfer_info}"
    )


def _update_med_and_check(
    *,
    model,
    med_replay_enabled: bool,
    current_batch: List[Dict[str, object]],
    med_seen_keys: set[tuple[str, str]] | None,
) -> int:
    """Update MED with current-only samples and verify seen-key growth."""
    if not med_replay_enabled:
        return 0

    seen_before = len(med_seen_keys) if med_seen_keys is not None else -1
    added = 0
    for rec in current_batch:
        if bool(rec.get("is_memory_sample", False)):
            raise RuntimeError("MED strict violation: replay samples cannot be written back to MED.")
        key = (str(rec["dataset_id"]), str(rec["image_id"]))
        if med_seen_keys is None or key not in med_seen_keys:
            model.update_med_from_source_text(
                source_text=str(rec["source_text"]),
                image_id=str(rec["image_id"]),
                dataset_id=str(rec["dataset_id"]),
            )
            if med_seen_keys is not None:
                med_seen_keys.add(key)
            added += 1
    seen_after = len(med_seen_keys) if med_seen_keys is not None else -1
    if med_seen_keys is not None and seen_after != seen_before + added:
        raise RuntimeError("MED seen-key growth mismatch: replay samples may have been written back unexpectedly")
    return added


def _run_semantic_train_step_merged(*, model, criterion, merged_batch: List[Dict[str, object]], snr_db: float):
    source_texts = [str(item["source_text"]) for item in merged_batch]
    out = model.forward_semantic_phase(
        image=None,
        snr_db=snr_db,
        source_text=source_texts,
    )
    _assert_semantic_forward_contract(out)
    loss = criterion(out["logits"].reshape(-1, out["logits"].size(-1)), out["target_ids"].reshape(-1))
    return out, loss


def _run_joint_train_step_merged(*, model, criterion, merged_batch: List[Dict[str, object]], snr_db: float):
    source_texts = [str(item["source_text"]) for item in merged_batch]
    out = model.forward_joint_phase(
        image=None,
        snr_db=snr_db,
        source_text=source_texts,
        image_id="merged_batch",
        dataset_id="merged",
    )
    _assert_semantic_forward_contract(out)
    return out


def _enrich_saved_checkpoints(
    checkpoint_paths: list,
    sender: str,
    snr_train_mode: str,
    seed: int,
    channel_type: str,
    use_nam: bool,
    enable_med: bool,
    caption_mode: str = "baseline",
) -> None:
    """在训练完成后为保存的 checkpoint 注入元数据 (任务书 §8.3)。"""
    import torch
    try:
        from audit.checkpoint_meta import enrich_checkpoint_dict, get_git_hash
    except ImportError:
        import sys
        _audit_dir = str(Path(__file__).resolve().parent.parent / "audit")
        if _audit_dir not in sys.path:
            sys.path.insert(0, _audit_dir)
        from checkpoint_meta import enrich_checkpoint_dict, get_git_hash

    for ckpt_path_str in checkpoint_paths:
        ckpt_path = Path(str(ckpt_path_str))
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "meta_git_hash" in ckpt:
            continue  # 已有元数据，跳过
        phase = ckpt.get("phase", "unknown") if isinstance(ckpt, dict) else "unknown"
        enrich_checkpoint_dict(
            ckpt,
            figure_name=ckpt.get("meta_figure_name", ""),
            experiment_name=f"{sender}_{phase}",
            dataset_split_id="train",
            sender_kb=sender,
            receiver_kb="sd",
            use_med=bool(enable_med),
            use_nam=bool(use_nam),
            channel_type=channel_type,
            train_snr_mode=snr_train_mode,
            seed=seed,
            phase=phase,
        )
        # 将 caption_mode 直接存入 checkpoint，供评估侧一致性验证
        ckpt["caption_mode"] = caption_mode
        torch.save(ckpt, ckpt_path)


def _compute_bleu_n(references: List[str], hypotheses: List[str], n: int) -> float:
    if len(references) == 0 or len(hypotheses) == 0:
        raise RuntimeError("Fig8 BLEU requires non-empty references and hypotheses.")
    if len(references) != len(hypotheses):
        raise RuntimeError("Fig8 BLEU input size mismatch between references and hypotheses.")

    bleu_module = importlib.import_module("nltk.translate.bleu_score")
    smoothing = getattr(bleu_module, "SmoothingFunction")().method1
    corpus_bleu = getattr(bleu_module, "corpus_bleu")

    refs = [[str(ref).strip().lower().split()] for ref in references]
    hyps = [str(hyp).strip().lower().split() for hyp in hypotheses]
    weights = (1.0, 0.0, 0.0, 0.0) if n == 1 else (0.5, 0.5, 0.0, 0.0)
    return float(corpus_bleu(refs, hyps, weights=weights, smoothing_function=smoothing))


def _evaluate_bleu_on_records(
    *,
    model,
    records: List[Dict],
    snr_db: float,
    sd_steps: int,
    max_batches: int,
    batch_size: int,
    seed: int,
) -> Dict[str, float]:
    references: List[str] = []
    hypotheses: List[str] = []
    batches = chunk_records(records, batch_size=batch_size, max_batches=max_batches)
    sample_index = 0
    for batch in batches:
        for rec in batch:
            image = Image.open(rec["path"]).convert("RGB")
            out = model.infer_full(
                image=image,
                snr_db=snr_db,
                sd_height=256,
                sd_width=256,
                sd_steps=sd_steps,
                sd_guidance=7.5,
                sd_seed=seed + sample_index,
                return_debug=False,
            )
            references.append(str(out["source_text"]))
            hypotheses.append(str(out["recovered_text"]))
            sample_index += 1

    return {
        "bleu1": _compute_bleu_n(references, hypotheses, n=1),
        "bleu2": _compute_bleu_n(references, hypotheses, n=2),
    }


def _write_matrix_csv(rows: List[Dict], out_csv: Path, score_field: str) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sender", "train_stage", "test_dataset", score_field, "checkpoint"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sender": row["sender"],
                    "train_stage": row["train_stage"],
                    "test_dataset": row["test_dataset"],
                    score_field: row[score_field],
                    "checkpoint": row["checkpoint"],
                }
            )


def _validate_fig8_variant_checkpoint_map_complete(
    checkpoint_map: Dict[str, Dict[str, Dict[str, str]]],
    *,
    variants: List[str],
    senders: List[str],
    tasks: List[str],
) -> None:
    if set(checkpoint_map.keys()) != set(variants):
        raise RuntimeError(
            f"Fig8 checkpoint map variants mismatch: expected={sorted(variants)}, got={sorted(checkpoint_map.keys())}"
        )

    for variant in variants:
        sender_map = checkpoint_map.get(variant)
        if not isinstance(sender_map, dict):
            raise RuntimeError(f"Fig8 checkpoint map variant block must be dict: {variant}")
        if set(sender_map.keys()) != set(senders):
            raise RuntimeError(
                f"Fig8 checkpoint map senders mismatch for variant={variant}: expected={sorted(senders)}, got={sorted(sender_map.keys())}"
            )

        for sender in senders:
            task_map = sender_map.get(sender)
            if not isinstance(task_map, dict):
                raise RuntimeError(f"Fig8 checkpoint map sender block must be dict: variant={variant}, sender={sender}")
            if set(task_map.keys()) != set(tasks):
                raise RuntimeError(
                    f"Fig8 checkpoint map tasks mismatch for variant={variant}, sender={sender}: "
                    f"expected={sorted(tasks)}, got={sorted(task_map.keys())}"
                )

            for task in tasks:
                ckpt = Path(str(task_map[task])).expanduser().resolve()
                if not ckpt.exists():
                    raise RuntimeError(
                        f"Fig8 checkpoint path missing for variant={variant}, sender={sender}, task={task}: {ckpt}"
                    )
