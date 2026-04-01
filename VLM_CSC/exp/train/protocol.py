"""三阶段论文训练协议 + 单 sender 完整训练流程。"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

from common.model_builder import build_vlm_system
from common.utils import chunk_records
from train.caption_cache import (
    ensure_captions_for_sender as _ensure_captions_for_sender,
)
from train.resume_manager import ResumeManager

from train.helpers import (
    _enrich_saved_checkpoints,
    _load_phase_best_checkpoint_strict,
)
from train.phases import run_channel_phase, run_semantic_phase, run_joint_phase


def run_paper_training_protocol(
    *,
    model,
    sender: str,
    train_batches,
    val_batches,
    caption_cache,
    rng,
    phase_cfg,
    snr_min_db,
    snr_max_db,
    snr_train_mode,
    checkpoint_dir: Path,
    dataset_id: str = "train",
    med_seen_keys: set[tuple[str, str]] | None = None,
    med_replay_enabled: bool = False,
    med_replay_batch_size: int = 4,
    med_replay_stm_ratio: float = 0.5,
    med_replay_weight: float = 1.0,
    resume_mgr: ResumeManager | None = None,
    resume_key: str = "",
):
    import torch

    criterion = torch.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_id)

    # ── Phase A: Channel ─────────────────────────────────────
    _ck_channel = checkpoint_dir / f"{sender}_phase_channel_best.pth"
    if resume_mgr and resume_mgr.is_phase_complete(resume_key, "channel") and _ck_channel.exists():
        print(f"[resume] >> 跳过 channel phase ({sender})")
        _load_phase_best_checkpoint_strict(model=model, checkpoint_path=_ck_channel, phase_name="phaseA_resume")
        phaseA = {"checkpoint": str(_ck_channel)}
    else:
        phaseA = run_channel_phase(
            model=model,
            sender=sender,
            train_batches=train_batches,
            val_batches=val_batches,
            caption_cache=caption_cache,
            rng=rng,
            phase_cfg=phase_cfg["channel_phase"],
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
            snr_train_mode=snr_train_mode,
            checkpoint_path=_ck_channel,
        )
        phaseA_best = Path(str(phaseA.get("checkpoint", "")))
        if str(phaseA_best) == "" or not phaseA_best.exists():
            raise RuntimeError(
                f"phaseA did not produce a valid best checkpoint for sender={sender}: {phaseA.get('checkpoint')}"
            )
        _load_phase_best_checkpoint_strict(model=model, checkpoint_path=phaseA_best, phase_name="phaseA")
        if resume_mgr and resume_key:
            resume_mgr.mark_phase_complete(resume_key, "channel", str(phaseA_best))

    # ── Phase B: Semantic ────────────────────────────────────
    _ck_semantic = checkpoint_dir / f"{sender}_phase_semantic_best.pth"
    if resume_mgr and resume_mgr.is_phase_complete(resume_key, "semantic") and _ck_semantic.exists():
        print(f"[resume] >> 跳过 semantic phase ({sender})")
        _load_phase_best_checkpoint_strict(model=model, checkpoint_path=_ck_semantic, phase_name="phaseB_resume")
        phaseB = {"checkpoint": str(_ck_semantic)}
    else:
        phaseB = run_semantic_phase(
            model=model,
            sender=sender,
            train_batches=train_batches,
            val_batches=val_batches,
            caption_cache=caption_cache,
            criterion=criterion,
            rng=rng,
            phase_cfg=phase_cfg["semantic_phase"],
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
            snr_train_mode=snr_train_mode,
            checkpoint_path=_ck_semantic,
            med_replay_enabled=med_replay_enabled,
            med_replay_batch_size=med_replay_batch_size,
            med_replay_stm_ratio=med_replay_stm_ratio,
            med_replay_weight=med_replay_weight,
            dataset_id=dataset_id,
            med_seen_keys=med_seen_keys,
        )
        phaseB_best = Path(str(phaseB.get("checkpoint", "")))
        if str(phaseB_best) == "" or not phaseB_best.exists():
            raise RuntimeError(
                f"phaseB did not produce a valid best checkpoint for sender={sender}: {phaseB.get('checkpoint')}"
            )
        _load_phase_best_checkpoint_strict(model=model, checkpoint_path=phaseB_best, phase_name="phaseB")
        if resume_mgr and resume_key:
            resume_mgr.mark_phase_complete(resume_key, "semantic", str(phaseB_best))

    # ── Phase C: Joint ───────────────────────────────────────
    _ck_joint_best = checkpoint_dir / f"{sender}_phase_joint_best.pth"
    _ck_joint_last = checkpoint_dir / f"{sender}_phase_joint_last.pth"
    if (resume_mgr and resume_mgr.is_phase_complete(resume_key, "joint_best")
            and _ck_joint_best.exists() and _ck_joint_last.exists()):
        print(f"[resume] >> 跳过 joint phase ({sender})")
        _load_phase_best_checkpoint_strict(model=model, checkpoint_path=_ck_joint_best, phase_name="phaseC_resume")
        phaseC = {"best_checkpoint": str(_ck_joint_best), "last_checkpoint": str(_ck_joint_last)}
    else:
        phaseC = run_joint_phase(
            model=model,
            sender=sender,
            train_batches=train_batches,
            val_batches=val_batches,
            caption_cache=caption_cache,
            criterion=criterion,
            rng=rng,
            phase_cfg=phase_cfg["joint_phase"],
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
            snr_train_mode=snr_train_mode,
            best_checkpoint_path=_ck_joint_best,
            last_checkpoint_path=_ck_joint_last,
            med_replay_enabled=med_replay_enabled,
            med_replay_batch_size=med_replay_batch_size,
            med_replay_stm_ratio=med_replay_stm_ratio,
            med_replay_weight=med_replay_weight,
            dataset_id=dataset_id,
            med_seen_keys=med_seen_keys,
        )
        phaseC_best = Path(str(phaseC.get("best_checkpoint", "")))
        phaseC_last = Path(str(phaseC.get("last_checkpoint", "")))
        if str(phaseC_best) == "" or not phaseC_best.exists():
            raise RuntimeError(
                f"phaseC did not produce a valid best checkpoint for sender={sender}: {phaseC.get('best_checkpoint')}"
            )
        if str(phaseC_last) == "" or not phaseC_last.exists():
            raise RuntimeError(
                f"phaseC did not produce a valid last checkpoint for sender={sender}: {phaseC.get('last_checkpoint')}"
            )
        _load_phase_best_checkpoint_strict(model=model, checkpoint_path=phaseC_best, phase_name="phaseC")
        if resume_mgr and resume_key:
            resume_mgr.mark_phase_complete(resume_key, "joint_best", str(phaseC_best))
            resume_mgr.mark_phase_complete(resume_key, "joint_last", str(phaseC_last))

    return phaseA, phaseB, phaseC


def train_sender(
    *,
    sender: str,
    train_records: List[Dict],
    val_records: List[Dict],
    vlm_module,
    blip_dir: Path,
    ram_ckpt: Path | None,
    sd_dir: Path,
    channel_type: str,
    device: str,
    quiet_third_party: bool,
    lr: float,
    weight_decay: float,
    snr_min_db: float,
    snr_max_db: float,
    snr_train_mode: str,
    batch_size: int,
    max_batches: int,
    use_caption_cache: bool,
    caption_cache_dir: Path,
    strict_cache_required: bool,
    seed: int,
    phase_cfg: Dict,
    checkpoint_dir: Path,
    enable_med: bool = False,
    med_kwargs: Dict | None = None,
    max_text_len: int = 24,
    max_text_len_by_sender: Dict[str, int] | None = None,
    med_replay_enabled: bool = False,
    med_replay_batch_size: int = 4,
    med_replay_stm_ratio: float = 0.5,
    med_replay_weight: float = 1.0,
    auto_rebuild_cache_on_hash_mismatch: bool = True,
    use_nam: bool = True,
    channel_dim: int | None = None,
    caption_mode: str = "baseline",
    caption_prompt: str | None = None,
    resume_mgr: ResumeManager | None = None,
    resume_key: str = "",
) -> Dict[str, object]:
    model = build_vlm_system(
        vlm_module,
        sender=sender,
        blip_dir=blip_dir,
        ram_ckpt=ram_ckpt,
        sd_dir=sd_dir,
        channel_type=channel_type,
        device=device,
        quiet_third_party=quiet_third_party,
        use_real_receiver_ckb=False,
        enable_med=enable_med,
        med_kwargs=med_kwargs,
        max_text_len=int(max_text_len),
        max_text_len_by_sender=max_text_len_by_sender,
        use_nam=bool(use_nam),
        channel_dim=channel_dim,
        caption_mode=caption_mode,
        caption_prompt=caption_prompt,
    )
    # --- Structural NAM assertion (anti-cheat) ---
    nam_param_count = sum(1 for n, _ in model.named_parameters() if "nam" in n.lower())
    if use_nam and nam_param_count == 0:
        raise RuntimeError(f"use_nam=True but model has 0 NAM parameters — structural integrity violation (sender={sender})")
    if (not use_nam) and nam_param_count > 0:
        raise RuntimeError(
            f"use_nam=False but model still has {nam_param_count} NAM parameters — "
            f"structural removal failed (sender={sender}). "
            f"This violates Fig9 ablation protocol: NAM must be structurally absent."
        )
    if bool(enable_med) and model.med is None:
        raise RuntimeError(f"enable_med=True but MED is not initialized for sender={sender}")
    if (not bool(enable_med)) and model.med is not None:
        raise RuntimeError(f"enable_med=False but MED is initialized for sender={sender}")

    train_batches = chunk_records(train_records, batch_size=batch_size, max_batches=max_batches)
    val_batches = chunk_records(val_records, batch_size=batch_size, max_batches=max_batches)
    if len(train_batches) == 0 or len(val_batches) == 0:
        raise RuntimeError(f"Insufficient train/val batches for sender={sender}")

    all_records = list(train_records) + list(val_records)
    # 每种 caption_mode 使用独立子目录，避免切换模式后缓存互相污染
    cache_file = (caption_cache_dir / caption_mode) / f"{sender}_captions.json"
    caption_cache = _ensure_captions_for_sender(
        model=model,
        sender=sender,
        records=all_records,
        cache_file=cache_file,
        use_caption_cache=use_caption_cache,
        strict_cache_required=strict_cache_required,
        auto_rebuild=auto_rebuild_cache_on_hash_mismatch,
        caption_prompt=caption_prompt,
        caption_mode=caption_mode,
    )

    phase_cfg = dict(phase_cfg)
    for phase_name in ["channel_phase", "semantic_phase", "joint_phase"]:
        phase_cfg[phase_name]["lr"] = phase_cfg[phase_name].get("lr", lr)
        phase_cfg[phase_name]["weight_decay"] = phase_cfg[phase_name].get("weight_decay", weight_decay)

    rng = random.Random(seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    phaseA, phaseB, phaseC = run_paper_training_protocol(
        model=model,
        sender=sender,
        train_batches=train_batches,
        val_batches=val_batches,
        caption_cache=caption_cache,
        rng=rng,
        phase_cfg=phase_cfg,
        snr_min_db=snr_min_db,
        snr_max_db=snr_max_db,
        snr_train_mode=snr_train_mode,
        checkpoint_dir=checkpoint_dir,
        med_replay_enabled=med_replay_enabled,
        med_replay_batch_size=med_replay_batch_size,
        med_replay_stm_ratio=med_replay_stm_ratio,
        med_replay_weight=med_replay_weight,
        resume_mgr=resume_mgr,
        resume_key=resume_key,
    )

    # ── 检查点元数据注入 (任务书 §8.3) ──
    _enrich_saved_checkpoints(
        checkpoint_paths=[
            phaseA["checkpoint"],
            phaseB["checkpoint"],
            phaseC["best_checkpoint"],
            phaseC["last_checkpoint"],
        ],
        sender=sender,
        snr_train_mode=snr_train_mode,
        seed=seed,
        channel_type=channel_type,
        use_nam=use_nam,
        enable_med=enable_med,
        caption_mode=caption_mode,
    )

    return {
        "sender": sender,
        "phase_channel_checkpoint": phaseA["checkpoint"],
        "phase_semantic_checkpoint": phaseB["checkpoint"],
        "phase_joint_best_checkpoint": phaseC["best_checkpoint"],
        "phase_joint_last_checkpoint": phaseC["last_checkpoint"],
        "checkpoint": phaseC["best_checkpoint"],
    }
