"""训练入口路由：_run_training_core + run_fig7/8/9/10_protocol + run_training。"""
from __future__ import annotations

import csv
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict

from common.utils import (
    collect_binary_images_from_split as collect_images_from_split,
    configure_runtime_logging,
    chunk_records,
)
from train.resume_manager import ResumeManager, prompt_resume_or_restart

from train.config import TrainConfig, set_seed, load_vlm_module, _validate_train_phase_config
from train.protocol import train_sender
from train.fig8_continual import run_fig8_continual_training


def _run_training_core(config: TrainConfig) -> Dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment, but no CUDA device is available.")

    device = "cuda"

    configure_runtime_logging(config.quiet_third_party)
    set_seed(config.seed)

    if config.strict_paper_repro:
        if config.channel_type not in {"awgn", "rayleigh"}:
            raise RuntimeError(f"Invalid channel_type under strict mode: {config.channel_type}")
        if config.train_snr_mode not in {"fixed_point", "uniform_range"}:
            raise RuntimeError(f"Invalid train_snr_mode under strict mode: {config.train_snr_mode}")
        if config.train_batch_size <= 0:
            raise RuntimeError("train_batch_size must be > 0 under strict mode.")

    phase_cfg = _validate_train_phase_config(config)

    is_fig8_continual_mode = (config.fig_name.lower() == "fig8") or (config.dataset_sequence is not None)
    if is_fig8_continual_mode:
        if str(config.train_split_dir).strip() != "":
            raise RuntimeError(
                "Fig8 strict continual mode does not allow legacy train_split_dir. "
                "Use dataset_sequence + dataset_splits only."
            )
        if config.dataset_sequence is None:
            raise RuntimeError("Fig8 continual training requires dataset_sequence in TrainConfig.")
        if config.dataset_sequence != ["cifar", "birds", "catsvsdogs"]:
            raise RuntimeError(
                "Fig8 continual training requires dataset_sequence=['cifar','birds','catsvsdogs']"
            )
        if config.dataset_splits is None:
            raise RuntimeError("Fig8 continual training requires dataset_splits in TrainConfig.")
        return run_fig8_continual_training(config=config, device=device)

    if config.strict_paper_repro and "fig9" in config.train_tag.lower():
        if "with_nam" in config.train_tag.lower():
            if not config.use_nam:
                raise RuntimeError("Fig9 with-NAM strict protocol requires use_nam=True")
            if config.train_snr_mode != "uniform_range":
                raise RuntimeError("Fig9 with-NAM strict protocol requires train_snr_mode=uniform_range")
            if abs(config.train_snr_min_db - 0.0) > 1e-12 or abs(config.train_snr_max_db - 10.0) > 1e-12:
                raise RuntimeError("Fig9 with-NAM strict protocol requires train_snr range [0, 10]")
        if "without_nam" in config.train_tag.lower():
            if config.use_nam:
                raise RuntimeError(
                    "Fig9 without-NAM strict protocol requires use_nam=False "
                    "(structural NAM removal, not just config naming)"
                )
            allowed = {0.0, 2.0, 4.0, 8.0}
            if config.train_snr_mode != "fixed_point":
                raise RuntimeError("Fig9 without-NAM strict protocol requires train_snr_mode=fixed_point")
            if float(config.train_snr_min_db) != float(config.train_snr_max_db):
                raise RuntimeError("Fig9 without-NAM strict protocol requires fixed single train SNR point")
            if float(config.train_snr_min_db) not in allowed:
                raise RuntimeError("Fig9 without-NAM strict protocol requires train_snr_db in {0,2,4,8}")

    project_root = Path(config.project_root)
    vlm_module = load_vlm_module(Path(config.model_file))
    train_split_dir = Path(config.train_split_dir)
    train_records_all = collect_images_from_split(train_split_dir, config.train_max_per_class)
    if len(train_records_all) == 0:
        raise RuntimeError(f"No train samples found in split: {train_split_dir}")

    rng = random.Random(config.seed)
    rng.shuffle(train_records_all)
    val_size = max(1, int(len(train_records_all) * config.val_ratio))
    if val_size >= len(train_records_all):
        val_size = max(1, len(train_records_all) // 5)
    val_records = train_records_all[:val_size]
    train_records = train_records_all[val_size:]
    if len(train_records) == 0:
        raise RuntimeError("Train split became empty after val split; increase train_max_per_class or reduce val_ratio.")

    checkpoint_dir = Path(config.checkpoint_dir)
    output_dir = Path(config.output_dir)
    caption_cache_dir = Path(config.caption_cache_dir) if config.caption_cache_dir else output_dir / "caption_cache"

    # ── 断点续传 ──────────────────────────────────────────────
    resume_mgr = ResumeManager(checkpoint_dir, config.fig_name or "default")
    prompt_resume_or_restart(resume_mgr)

    results = []
    for sender in config.senders:
        resume_key = sender
        # 如果该 sender 的所有阶段均已完成，直接跳过
        if resume_mgr.is_fully_complete(resume_key):
            phases = resume_mgr.get_completed_phases(resume_key)
            print(f"\n[resume] >> 跳过 sender={sender}（已完成全部阶段）")
            results.append({
                "sender": sender,
                "phase_channel_checkpoint": phases.get("channel", ""),
                "phase_semantic_checkpoint": phases.get("semantic", ""),
                "phase_joint_best_checkpoint": phases.get("joint_best", ""),
                "phase_joint_last_checkpoint": phases.get("joint_last", ""),
                "checkpoint": phases.get("joint_best", ""),
            })
            continue
        result = train_sender(
            sender=sender,
            train_records=train_records,
            val_records=val_records,
            vlm_module=vlm_module,
            blip_dir=Path(config.blip_ckb_dir),
            ram_ckpt=Path(config.ram_ckb_path) if config.ram_ckb_path else None,
            sd_dir=Path(config.sd_ckb_dir),
            channel_type=config.channel_type,
            device=device,
            quiet_third_party=config.quiet_third_party,
            lr=config.train_lr,
            weight_decay=config.train_weight_decay,
            snr_min_db=config.train_snr_min_db,
            snr_max_db=config.train_snr_max_db,
            snr_train_mode=config.train_snr_mode,
            batch_size=config.train_batch_size,
            max_batches=config.train_max_batches,
            use_caption_cache=config.use_caption_cache,
            caption_cache_dir=caption_cache_dir,
            strict_cache_required=config.strict_cache_required,
            seed=config.seed,
            phase_cfg=phase_cfg,
            checkpoint_dir=checkpoint_dir,
            max_text_len=int(config.max_text_len),
            max_text_len_by_sender=config.max_text_len_by_sender,
            auto_rebuild_cache_on_hash_mismatch=bool(config.auto_rebuild_cache_on_hash_mismatch),
            use_nam=bool(config.use_nam),
            channel_dim=config.channel_dim,
            caption_mode=config.caption_mode,
            caption_prompt=config.caption_prompt,
            resume_mgr=resume_mgr,
            resume_key=resume_key,
        )
        results.append(result)

    # ── 训练完成，清理 resume 状态 ──
    resume_mgr.clear()

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"train_{config.train_tag}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sender",
                "phase_channel_checkpoint",
                "phase_semantic_checkpoint",
                "phase_joint_best_checkpoint",
                "phase_joint_last_checkpoint",
                "checkpoint",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    return {
        "summary_csv": str(summary_path),
        "results": results,
        "train_size": len(train_records),
        "val_size": len(val_records),
        "project_root": str(project_root),
    }


def run_fig7_protocol(config: TrainConfig) -> Dict[str, object]:
    cfg = replace(config, fig_name="fig7")
    if str(cfg.train_split_dir).strip() == "":
        raise RuntimeError("Fig7 protocol requires train_split_dir.")
    return _run_training_core(cfg)


def run_fig8_protocol(config: TrainConfig) -> Dict[str, object]:
    cfg = replace(config, fig_name="fig8")
    return _run_training_core(cfg)


def run_fig9_protocol(config: TrainConfig) -> Dict[str, object]:
    cfg = replace(config, fig_name="fig9")
    return _run_training_core(cfg)


def run_fig10_protocol(config: TrainConfig) -> Dict[str, object]:
    cfg = replace(config, fig_name="fig10")
    return _run_training_core(cfg)


def run_training(config: TrainConfig) -> Dict[str, object]:
    fig_name = str(config.fig_name).strip().lower()
    if fig_name == "fig8":
        return run_fig8_protocol(config)
    if fig_name == "fig9":
        return run_fig9_protocol(config)
    if fig_name == "fig10":
        return run_fig10_protocol(config)
    return run_fig7_protocol(config)
