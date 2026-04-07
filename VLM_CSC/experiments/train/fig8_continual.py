"""Fig8 持续学习训练：variant×sender×task 三层循环 + BLEU 监控。"""
from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

from common import (
    TaskDatasetManager,
    assert_fig8_variant_model_state,
    build_vlm_system,
    chunk_records,
    resolve_fig8_variant_med_config,
)
from train.caption_cache import (
    ensure_captions_for_sender as _ensure_captions_for_sender,
)
from train.resume_manager import ResumeManager, prompt_resume_or_restart

from train.config import TrainConfig, load_vlm_module, _validate_train_phase_config
from train.helpers import (
    _evaluate_bleu_on_records,
    _load_phase_best_checkpoint_strict,
    _validate_fig8_variant_checkpoint_map_complete,
    _write_matrix_csv,
)
from train.protocol import run_paper_training_protocol
from eval.metrics import plot_matrix_heatmap as _plot_matrix_heatmap


def _generate_train_monitor_heatmaps(
    *,
    bleu1_rows: List[Dict],
    bleu2_rows: List[Dict],
    dataset_sequence: List[str],
    variant: str,
    sender: str,
    output_dir: Path,
) -> None:
    """Build 3x3 matrices from BLEU rows and generate heatmaps for training monitor."""
    n = len(dataset_sequence)
    for score_field, rows, metric_label in [
        ("bleu1", bleu1_rows, "BLEU-1"),
        ("bleu2", bleu2_rows, "BLEU-2"),
    ]:
        matrix = np.zeros((n, n), dtype=float)
        for row in rows:
            stage_idx = dataset_sequence.index(row["train_stage"])
            test_idx = dataset_sequence.index(row["test_dataset"])
            matrix[stage_idx, test_idx] = float(row[score_field])
        _plot_matrix_heatmap(
            matrix,
            row_labels=dataset_sequence,
            col_labels=dataset_sequence,
            out_png=output_dir / f"fig8_{variant}_{sender}_train_monitor_val_{score_field}_heatmap.png",
            title=f"FIG8 {variant} {sender.upper()} {metric_label} Train Monitor (val)",
        )


def run_fig8_continual_training(config: TrainConfig, device: str) -> Dict[str, object]:
    import torch

    required_sequence = ["cifar", "birds", "catsvsdogs"]
    if config.dataset_sequence != required_sequence:
        raise RuntimeError(
            f"Fig8 continual training requires dataset_sequence={required_sequence}, got {config.dataset_sequence}"
        )
    if config.dataset_splits is None:
        raise RuntimeError("Fig8 continual training requires explicit dataset_splits with train/val/test roots per task.")
    if set(config.dataset_splits.keys()) != set(required_sequence):
        raise RuntimeError(
            f"Fig8 continual training requires dataset_splits keys exactly {required_sequence}, got {sorted(config.dataset_splits.keys())}"
        )

    project_root = Path(config.project_root)
    vlm_module = load_vlm_module(Path(config.model_file))
    phase_cfg = _validate_train_phase_config(config)

    output_dir = Path(config.output_dir)
    checkpoint_dir = Path(config.checkpoint_dir)
    caption_cache_root = Path(config.caption_cache_dir) if config.caption_cache_dir else output_dir / "caption_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    caption_cache_root.mkdir(parents=True, exist_ok=True)

    audit_manager = TaskDatasetManager(
        sequence=config.dataset_sequence,
        dataset_roots=config.dataset_roots,
        dataset_splits=config.dataset_splits,
        max_per_class=config.train_max_per_class,
        val_split_ratio=config.val_split_ratio,
        val_split_seed=config.val_split_seed,
        strict_mode=True,
        consumer="audit",
    )
    for task_name in required_sequence:
        summary = audit_manager.get_task_split_summary(task_name)
        train_paths = summary["train_paths"]
        val_paths = summary["val_paths"]
        test_paths = summary["test_paths"]
        if not train_paths.isdisjoint(val_paths):
            raise RuntimeError(f"Fig8 split leakage: train/val overlap for task={task_name}")
        if not train_paths.isdisjoint(test_paths):
            raise RuntimeError(f"Fig8 split leakage: train/test overlap for task={task_name}")
        if not val_paths.isdisjoint(test_paths):
            raise RuntimeError(f"Fig8 split leakage: val/test overlap for task={task_name}")

    task_manager = TaskDatasetManager(
        sequence=config.dataset_sequence,
        dataset_roots=config.dataset_roots,
        dataset_splits=config.dataset_splits,
        max_per_class=config.train_max_per_class,
        val_split_ratio=config.val_split_ratio,
        val_split_seed=config.val_split_seed,
        strict_mode=True,
        consumer="train",
    )

    if not config.senders:
        raise RuntimeError("Fig8 continual training requires non-empty senders")
    unsupported = [s for s in config.senders if s not in {"blip", "ram"}]
    if unsupported:
        raise RuntimeError(f"Fig8 continual training only supports senders in ['blip','ram'], got unsupported={unsupported}")

    variant_plan = ["with_med", "without_med"]
    med_kwargs_base = dict(config.med_kwargs) if config.med_kwargs is not None else None
    if med_kwargs_base is not None and "strict_paper_repro" not in med_kwargs_base:
        med_kwargs_base["strict_paper_repro"] = bool(config.strict_paper_repro)

    default_replay_batch = max(1, int(config.train_batch_size // 2))
    joint_cfg = dict(phase_cfg.get("joint_phase", {}))
    med_replay_batch_size = int(joint_cfg.get("med_replay_batch_size", default_replay_batch))
    med_replay_stm_ratio = float(joint_cfg.get("med_replay_stm_ratio", 0.5))
    med_replay_weight = float(joint_cfg.get("med_replay_weight", 1.0))

    bleu1_rows_all: List[Dict] = []
    bleu2_rows_all: List[Dict] = []
    task_checkpoint_rows: List[Dict] = []
    variant_checkpoint_map: Dict[str, Dict[str, Dict[str, str]]] = {
        variant: {sender: {} for sender in config.senders}
        for variant in ["with_med", "without_med"]
    }

    # ── 断点续传 ──────────────────────────────────────────────
    resume_mgr = ResumeManager(checkpoint_dir, "fig8")
    prompt_resume_or_restart(resume_mgr)

    for variant_name in variant_plan:
        enable_med, variant_med_kwargs = resolve_fig8_variant_med_config(
            variant=variant_name,
            med_kwargs_base=med_kwargs_base,
        )
        variant_bleu1_rows: List[Dict] = []
        variant_bleu2_rows: List[Dict] = []

        for sender in config.senders:
            sender_bleu1_rows: List[Dict] = []
            sender_bleu2_rows: List[Dict] = []
            med_seen_keys: set[tuple[str, str]] = set()
            model = build_vlm_system(
                vlm_module,
                sender=sender,
                blip_dir=Path(config.blip_ckb_dir),
                ram_ckpt=Path(config.ram_ckb_path),
                sd_dir=Path(config.sd_ckb_dir),
                channel_type=config.channel_type,
                device=device,
                quiet_third_party=config.quiet_third_party,
                use_real_receiver_ckb=False,
                enable_med=bool(enable_med),
                med_kwargs=variant_med_kwargs,
                max_text_len=int(config.max_text_len),
                max_text_len_by_sender=config.max_text_len_by_sender,
                channel_dim=config.channel_dim,
                caption_mode=config.caption_mode,
                caption_prompt=config.caption_prompt,
            )
            assert_fig8_variant_model_state(
                variant=variant_name,
                enable_med=bool(enable_med),
                med_kwargs=variant_med_kwargs,
                model=model,
            )

            for task_idx, task_name in enumerate(required_sequence):
                fig8_resume_key = f"{variant_name}/{sender}/{task_name}"

                # ── 断点续传：如果该 task 的全部 phase 已完成，直接跳过训练 ──
                stage_checkpoint = checkpoint_dir / variant_name / f"{sender}_{task_idx + 1}_{task_name}_phase_joint_best.pth"
                if resume_mgr.is_fully_complete(fig8_resume_key) and stage_checkpoint.exists():
                    print(f"[resume] >> 跳过 {fig8_resume_key}（已完成）")
                    _load_phase_best_checkpoint_strict(
                        model=model, checkpoint_path=stage_checkpoint,
                        phase_name=f"resume_{fig8_resume_key}",
                    )
                    variant_checkpoint_map[variant_name][sender][task_name] = str(stage_checkpoint)
                    task_checkpoint_rows.append({
                        "variant": variant_name,
                        "task_index": task_idx + 1,
                        "task_name": task_name,
                        "sender": sender,
                        "checkpoint": str(stage_checkpoint),
                        "phase_channel_checkpoint": resume_mgr.get_completed_phases(fig8_resume_key).get("channel", ""),
                        "phase_semantic_checkpoint": resume_mgr.get_completed_phases(fig8_resume_key).get("semantic", ""),
                        "phase_joint_last_checkpoint": resume_mgr.get_completed_phases(fig8_resume_key).get("joint_last", ""),
                    })
                else:
                    train_records = task_manager.get_task_train_set(task_name)
                    val_records = task_manager.get_task_val_set(task_name)

                    # Fig8 是 BLEU 文本保真度实验，非分类任务。
                    # 对非猫狗数据集将 label 置为 -1，避免训练阶段打印无意义的 A/B 分类评分。
                    if task_name != "catsvsdogs":
                        for _rec in train_records:
                            _rec["label"] = -1
                        for _rec in val_records:
                            _rec["label"] = -1

                    train_batches = chunk_records(train_records, batch_size=config.train_batch_size, max_batches=config.train_max_batches)
                    val_batches = chunk_records(val_records, batch_size=config.train_batch_size, max_batches=config.train_max_batches)
                    if len(train_batches) == 0 or len(val_batches) == 0:
                        raise RuntimeError(
                            f"Insufficient train/val batches for variant={variant_name}, sender={sender}, task={task_name}"
                        )

                    # caption 只取决于 sender + dataset + caption_mode，与 with_med/without_med 无关
                    # 统一放在 sender/task_N 下，跨 variant 复用
                    task_cache_dir = caption_cache_root / sender / f"task_{task_idx + 1}_{task_name}"
                    all_records = list(train_records) + list(val_records)
                    cache_file = task_cache_dir / f"{sender}_captions.json"

                    # 如果新路径没缓存，从旧 variant 目录迁移
                    if not cache_file.exists():
                        for _old_var in ["with_med", "without_med"]:
                            _old = caption_cache_root / _old_var / sender / f"task_{task_idx + 1}_{task_name}" / f"{sender}_captions.json"
                            if _old.exists():
                                cache_file.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(_old, cache_file)
                                print(f"[CAPTION_CACHE] 从 {_old_var}/ 迁移已有缓存 -> {cache_file.name}")
                                break

                    caption_cache = _ensure_captions_for_sender(
                        model=model,
                        sender=sender,
                        records=all_records,
                        cache_file=cache_file,
                        use_caption_cache=config.use_caption_cache,
                        strict_cache_required=config.strict_cache_required,
                        auto_rebuild=bool(config.auto_rebuild_cache_on_hash_mismatch),
                        caption_prompt=getattr(config, "caption_prompt", None),
                        caption_mode=getattr(config, "caption_mode", None),
                    )

                    task_phase_ckpt_dir = checkpoint_dir / variant_name / sender / f"task_{task_idx + 1}_{task_name}"
                    task_phase_ckpt_dir.mkdir(parents=True, exist_ok=True)
                    phaseA, phaseB, phaseC = run_paper_training_protocol(
                        model=model,
                        sender=sender,
                        train_batches=train_batches,
                        val_batches=val_batches,
                        caption_cache=caption_cache,
                        rng=random.Random(config.seed + task_idx),
                        phase_cfg=phase_cfg,
                        snr_min_db=config.train_snr_min_db,
                        snr_max_db=config.train_snr_max_db,
                        snr_train_mode=config.train_snr_mode,
                        checkpoint_dir=task_phase_ckpt_dir,
                        dataset_id=task_name,
                        med_seen_keys=med_seen_keys,
                        med_replay_enabled=bool(enable_med),
                        med_replay_batch_size=med_replay_batch_size,
                        med_replay_stm_ratio=med_replay_stm_ratio,
                        med_replay_weight=med_replay_weight,
                        resume_mgr=resume_mgr,
                        resume_key=fig8_resume_key,
                    )

                    stage_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(phaseC["best_checkpoint"], stage_checkpoint)
                    _load_phase_best_checkpoint_strict(model=model, checkpoint_path=stage_checkpoint, phase_name=f"{variant_name}:{sender}:{task_name}")

                    variant_checkpoint_map[variant_name][sender][task_name] = str(stage_checkpoint)
                    task_checkpoint_rows.append(
                        {
                            "variant": variant_name,
                            "task_index": task_idx + 1,
                            "task_name": task_name,
                            "sender": sender,
                            "checkpoint": str(stage_checkpoint),
                            "phase_channel_checkpoint": str(phaseA["checkpoint"]),
                            "phase_semantic_checkpoint": str(phaseB["checkpoint"]),
                            "phase_joint_last_checkpoint": str(phaseC["last_checkpoint"]),
                        }
                    )

                # ── 累积 BLEU 评估（无论训练还是 resume 都执行）──
                model.eval()
                seen_tasks = required_sequence[: task_idx + 1]
                for seen_dataset in seen_tasks:
                    if task_manager.consumer != "train":
                        raise RuntimeError("Fig8 train monitor requires TaskDatasetManager consumer='train'.")
                    bleu_scores = _evaluate_bleu_on_records(
                        model=model,
                        records=task_manager.get_task_val_set(seen_dataset),
                        snr_db=float(config.train_snr_min_db),
                        sd_steps=int(config.sd_steps),
                        max_batches=config.train_max_batches,
                        batch_size=config.train_batch_size,
                        seed=config.seed + task_idx,
                    )
                    row_base = {
                        "variant": variant_name,
                        "sender": sender,
                        "train_stage": task_name,
                        "test_dataset": seen_dataset,
                        "checkpoint": str(stage_checkpoint),
                    }
                    bleu1_row = dict(row_base)
                    bleu1_row["bleu1"] = float(bleu_scores["bleu1"])
                    bleu2_row = dict(row_base)
                    bleu2_row["bleu2"] = float(bleu_scores["bleu2"])
                    variant_bleu1_rows.append(bleu1_row)
                    variant_bleu2_rows.append(bleu2_row)
                    sender_bleu1_rows.append(bleu1_row)
                    sender_bleu2_rows.append(bleu2_row)
                    bleu1_rows_all.append(bleu1_row)
                    bleu2_rows_all.append(bleu2_row)

            _write_matrix_csv(
                sender_bleu1_rows,
                output_dir / f"fig8_{variant_name}_{sender}_train_monitor_val_bleu1_matrix.csv",
                score_field="bleu1",
            )
            _write_matrix_csv(
                sender_bleu2_rows,
                output_dir / f"fig8_{variant_name}_{sender}_train_monitor_val_bleu2_matrix.csv",
                score_field="bleu2",
            )

            # ── 训练监控热力图 ──
            _generate_train_monitor_heatmaps(
                bleu1_rows=sender_bleu1_rows,
                bleu2_rows=sender_bleu2_rows,
                dataset_sequence=required_sequence,
                variant=variant_name,
                sender=sender,
                output_dir=output_dir,
            )

        _write_matrix_csv(
            variant_bleu1_rows,
            output_dir / f"fig8_{variant_name}_train_monitor_val_bleu1_matrix.csv",
            score_field="bleu1",
        )
        _write_matrix_csv(
            variant_bleu2_rows,
            output_dir / f"fig8_{variant_name}_train_monitor_val_bleu2_matrix.csv",
            score_field="bleu2",
        )

    final_bleu1_csv = output_dir / "fig8_train_monitor_val_bleu1_matrix.csv"
    final_bleu2_csv = output_dir / "fig8_train_monitor_val_bleu2_matrix.csv"
    _write_matrix_csv(bleu1_rows_all, final_bleu1_csv, score_field="bleu1")
    _write_matrix_csv(bleu2_rows_all, final_bleu2_csv, score_field="bleu2")

    checkpoint_summary_csv = output_dir / "fig8_continual_task_checkpoints.csv"
    with checkpoint_summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "task_index",
                "task_name",
                "sender",
                "checkpoint",
                "phase_channel_checkpoint",
                "phase_semantic_checkpoint",
                "phase_joint_last_checkpoint",
            ],
        )
        writer.writeheader()
        for row in task_checkpoint_rows:
            writer.writerow(row)

    _validate_fig8_variant_checkpoint_map_complete(
        variant_checkpoint_map,
        variants=["with_med", "without_med"],
        senders=config.senders,
        tasks=required_sequence,
    )

    checkpoint_map_json = output_dir / "fig8_variant_checkpoint_map.json"
    checkpoint_map_json.write_text(json.dumps(variant_checkpoint_map, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 汇总最终 BLEU 指标到 summary ──
    def _lookup_final_bleu(rows: List[Dict], variant: str, sender: str) -> Dict[str, float]:
        """从 bleu_rows 中提取最后一个 train_stage 的各 test_dataset 最终 BLEU。"""
        final_stage = required_sequence[-1]
        out: Dict[str, float] = {}
        for r in rows:
            if r["variant"] == variant and r["sender"] == sender and r["train_stage"] == final_stage:
                ds = r["test_dataset"]
                score_key = "bleu1" if "bleu1" in r else "bleu2"
                out[ds] = float(r[score_key])
        return out

    final_summary_csv = output_dir / f"train_{config.train_tag}_summary.csv"
    with final_summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant", "sender", "checkpoint",
                "final_bleu1_cifar", "final_bleu1_birds", "final_bleu1_catsvsdogs",
                "final_bleu2_cifar", "final_bleu2_birds", "final_bleu2_catsvsdogs",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for variant_name in ["with_med", "without_med"]:
            for sender in config.senders:
                ckpt = variant_checkpoint_map[variant_name][sender][required_sequence[-1]]
                b1 = _lookup_final_bleu(bleu1_rows_all, variant_name, sender)
                b2 = _lookup_final_bleu(bleu2_rows_all, variant_name, sender)
                writer.writerow({
                    "variant": variant_name,
                    "sender": sender,
                    "checkpoint": ckpt,
                    "final_bleu1_cifar": b1.get("cifar", ""),
                    "final_bleu1_birds": b1.get("birds", ""),
                    "final_bleu1_catsvsdogs": b1.get("catsvsdogs", ""),
                    "final_bleu2_cifar": b2.get("cifar", ""),
                    "final_bleu2_birds": b2.get("birds", ""),
                    "final_bleu2_catsvsdogs": b2.get("catsvsdogs", ""),
                })

    # ── 训练完成，清理 resume 状态 ──
    resume_mgr.clear()

    return {
        "summary_csv": str(final_summary_csv),
        "task_checkpoint_csv": str(checkpoint_summary_csv),
        "checkpoint_map_json": str(checkpoint_map_json),
        "fig8_variant_checkpoint_map": variant_checkpoint_map,
        "final_bleu1_csv": str(final_bleu1_csv),
        "final_bleu2_csv": str(final_bleu2_csv),
        "results": task_checkpoint_rows,
        "project_root": str(project_root),
    }
