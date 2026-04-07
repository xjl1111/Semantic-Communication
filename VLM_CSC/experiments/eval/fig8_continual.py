"""Fig8 持续学习评估：BLEU 矩阵热力图。"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from common import (
    TaskDatasetManager,
    assert_fig8_variant_model_state,
    build_vlm_system,
    resolve_fig8_variant_med_config,
)
from eval.config import EvalConfig
from eval.validators import _get_fig8_variant_checkpoint, _normalize_fig8_med_variants
from eval.metrics import (
    evaluate_bleu_sender_snr,
    plot_matrix_heatmap as _plot_matrix_heatmap,
)


def _run_continual_bleu_map(
    config: EvalConfig,
    vlm_module,
    target_module,
    blip_dir: Path,
    ram_ckpt: Path,
    sd_dir: Path,
) -> Dict[str, str]:
    import torch

    if config.fig_name != "fig8":
        raise RuntimeError(f"Continual learning map mode is restricted to fig8, got fig_name={config.fig_name}")
    if config.channel_type != "rayleigh":
        raise RuntimeError("Fig8 strict protocol requires channel_type='rayleigh'")
    if config.metrics != ["bleu1", "bleu2"]:
        raise RuntimeError("Fig8 strict protocol requires metrics=['bleu1', 'bleu2']")
    if config.dataset_sequence != ["cifar", "birds", "catsvsdogs"]:
        raise RuntimeError("Fig8 strict protocol requires dataset_sequence=['cifar','birds','catsvsdogs']")
    variant_order = _normalize_fig8_med_variants(config.med_variants)
    if config.eval_output_mode != "continual_learning_map":
        raise RuntimeError("Fig8 strict protocol requires eval_output_mode='continual_learning_map'")

    if config.dataset_sequence is None or len(config.dataset_sequence) == 0:
        raise RuntimeError("Fig8 continual mode requires non-empty dataset_sequence")
    if config.dataset_roots is None:
        raise RuntimeError("Fig8 continual mode requires dataset_roots")
    if set(config.dataset_roots.keys()) != set(config.dataset_sequence):
        raise RuntimeError(
            f"Fig8 continual mode requires dataset_roots keys to match dataset_sequence, got keys={sorted(config.dataset_roots.keys())}"
        )
    if config.fig8_eval_snr_db is None:
        raise RuntimeError("Fig8 continual mode requires fig8_eval_snr_db (single numeric value).")
    if isinstance(config.fig8_eval_snr_db, (list, tuple, dict)):
        raise RuntimeError("fig8_eval_snr_db must be a single numeric value, not a collection.")

    device = "cuda"
    snr_db = float(config.fig8_eval_snr_db)
    if not np.isfinite(snr_db):
        raise RuntimeError(f"fig8_eval_snr_db must be finite, got {config.fig8_eval_snr_db}")
    output_dir = Path(config.output_dir)
    if "train_monitor" in [p.lower() for p in output_dir.parts]:
        raise RuntimeError("Fig8 final evaluation must write to final_eval directory, not train_monitor.")
    output_dir.mkdir(parents=True, exist_ok=True)

    task_manager = TaskDatasetManager(
        sequence=config.dataset_sequence,
        dataset_roots=config.dataset_roots,
        dataset_splits=config.dataset_splits,
        max_per_class=config.max_per_class,
        strict_mode=True,
        consumer="eval",
    )
    detail_rows: List[Dict] = []

    med_kwargs_base = dict(config.med_kwargs) if config.med_kwargs is not None else None
    if med_kwargs_base is not None and "strict_paper_repro" not in med_kwargs_base:
        med_kwargs_base["strict_paper_repro"] = bool(config.strict_paper_repro)

    for variant in variant_order:
        if variant == "with_med":
            if config.fig8_variant_checkpoint_map is None or variant not in config.fig8_variant_checkpoint_map:
                raise RuntimeError("with_med=True but MED trajectory checkpoints are missing")

        enable_med, variant_med_kwargs = resolve_fig8_variant_med_config(
            variant=variant,
            med_kwargs_base=med_kwargs_base,
        )

        variant_bleu1_rows = []
        variant_bleu2_rows = []
        for sender in config.senders:
            sender_matrix_bleu1 = np.zeros((len(config.dataset_sequence), len(config.dataset_sequence)), dtype=float)
            sender_matrix_bleu2 = np.zeros((len(config.dataset_sequence), len(config.dataset_sequence)), dtype=float)
            sender_bleu1_rows = []
            sender_bleu2_rows = []

            for stage_idx, stage_dataset in enumerate(config.dataset_sequence):
                model = build_vlm_system(
                    vlm_module,
                    sender=sender,
                    blip_dir=blip_dir,
                    ram_ckpt=ram_ckpt,
                    sd_dir=sd_dir,
                    channel_type=config.channel_type,
                    device=device,
                    quiet_third_party=config.quiet_third_party,
                    use_real_receiver_ckb=False,  # BLEU 只比较文本，无需加载 SD (~4GB VRAM)
                    enable_med=bool(enable_med),
                    med_kwargs=variant_med_kwargs,
                    max_text_len=int(config.max_text_len),
                    max_text_len_by_sender=config.max_text_len_by_sender,
                    channel_dim=config.channel_dim,
                    caption_mode=config.caption_mode,
                    caption_prompt=config.caption_prompt,
                )
                assert_fig8_variant_model_state(
                    variant=variant,
                    enable_med=bool(enable_med),
                    med_kwargs=variant_med_kwargs,
                    model=model,
                )

                ckpt_path = _get_fig8_variant_checkpoint(config, variant=variant, sender=sender, stage_dataset=stage_dataset)
                seen_tasks = task_manager.get_seen_tasks(stage_idx)
                state = torch.load(ckpt_path, map_location=device)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=True)
                model.eval()

                for test_idx, test_dataset in enumerate(seen_tasks):
                    split_name = "test"
                    if split_name != "test":
                        raise RuntimeError("Fig8 final eval split must be test.")
                    records = task_manager.get_task_test_set(test_dataset)
                    if len(records) == 0:
                        raise RuntimeError(f"No records for continual eval dataset={test_dataset}")

                    bleu_scores, rows = evaluate_bleu_sender_snr(
                        model=model,
                        sender_type=sender,
                        snr_db=snr_db,
                        records=records,
                        target_module=target_module,
                        sd_steps=config.sd_steps,
                        sd_height=config.sd_height,
                        sd_width=config.sd_width,
                        batch_size=config.batch_size,
                        max_batches=config.max_batches,
                        seed=config.seed,
                    )
                    sender_matrix_bleu1[stage_idx, test_idx] = float(bleu_scores["bleu1"])
                    sender_matrix_bleu2[stage_idx, test_idx] = float(bleu_scores["bleu2"])

                    detail_rows.extend(rows)
                    variant_bleu1_rows.append(
                        {
                            "variant": variant,
                            "sender": sender,
                            "train_stage": stage_dataset,
                            "test_dataset": test_dataset,
                            "bleu1": float(bleu_scores["bleu1"]),
                            "checkpoint": str(ckpt_path),
                        }
                    )
                    sender_bleu1_rows.append(
                        {
                            "variant": variant,
                            "sender": sender,
                            "train_stage": stage_dataset,
                            "test_dataset": test_dataset,
                            "bleu1": float(bleu_scores["bleu1"]),
                            "checkpoint": str(ckpt_path),
                        }
                    )
                    variant_bleu2_rows.append(
                        {
                            "variant": variant,
                            "sender": sender,
                            "train_stage": stage_dataset,
                            "test_dataset": test_dataset,
                            "bleu2": float(bleu_scores["bleu2"]),
                            "checkpoint": str(ckpt_path),
                        }
                    )
                    sender_bleu2_rows.append(
                        {
                            "variant": variant,
                            "sender": sender,
                            "train_stage": stage_dataset,
                            "test_dataset": test_dataset,
                            "bleu2": float(bleu_scores["bleu2"]),
                            "checkpoint": str(ckpt_path),
                        }
                    )

            _plot_matrix_heatmap(
                sender_matrix_bleu1,
                row_labels=config.dataset_sequence,
                col_labels=config.dataset_sequence,
                out_png=output_dir / f"fig8_{variant}_{sender}_final_test_bleu1_heatmap_snr{snr_db:g}.png",
                title=f"FIG8 {variant} {sender.upper()} BLEU-1 Continual Map",
            )
            _plot_matrix_heatmap(
                sender_matrix_bleu2,
                row_labels=config.dataset_sequence,
                col_labels=config.dataset_sequence,
                out_png=output_dir / f"fig8_{variant}_{sender}_final_test_bleu2_heatmap_snr{snr_db:g}.png",
                title=f"FIG8 {variant} {sender.upper()} BLEU-2 Continual Map",
            )

            sender_bleu1_csv = output_dir / f"fig8_{variant}_{sender}_final_test_bleu1_matrix.csv"
            sender_bleu2_csv = output_dir / f"fig8_{variant}_{sender}_final_test_bleu2_matrix.csv"
            with sender_bleu1_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["variant", "sender", "train_stage", "test_dataset", "bleu1", "checkpoint"])
                writer.writeheader()
                for row in sender_bleu1_rows:
                    writer.writerow(row)
            with sender_bleu2_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["variant", "sender", "train_stage", "test_dataset", "bleu2", "checkpoint"])
                writer.writeheader()
                for row in sender_bleu2_rows:
                    writer.writerow(row)

        bleu1_csv = output_dir / f"fig8_{variant}_final_test_bleu1_matrix.csv"
        bleu2_csv = output_dir / f"fig8_{variant}_final_test_bleu2_matrix.csv"
        with bleu1_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["variant", "sender", "train_stage", "test_dataset", "bleu1", "checkpoint"])
            writer.writeheader()
            for row in variant_bleu1_rows:
                writer.writerow(row)
        with bleu2_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["variant", "sender", "train_stage", "test_dataset", "bleu2", "checkpoint"])
            writer.writeheader()
            for row in variant_bleu2_rows:
                writer.writerow(row)

    detail_csv = output_dir / f"fig8_{config.tag}_final_test_details.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sender", "snr_db", "image", "source_text", "recovered_text"], extrasaction="ignore")
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)

    summary_txt = output_dir / f"fig8_{config.tag}_final_test_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write(f"dataset_sequence: {config.dataset_sequence}\n")
        f.write(f"fig8_eval_snr_db: {snr_db}\n")
        f.write(f"training_snr_protocol: {config.training_snr_protocol}\n")
        f.write(f"metrics: {config.metrics}\n")
        f.write(f"med_variants: {config.med_variants}\n")
        f.write(f"eval_output_mode: {config.eval_output_mode}\n")

    return {
        "detail_csv": str(detail_csv),
        "curve_csv": str(output_dir / f"fig8_with_med_{config.senders[0]}_final_test_bleu1_matrix.csv"),
        "plot_png": str(output_dir / f"fig8_with_med_{config.senders[0]}_final_test_bleu1_heatmap_snr{snr_db:g}.png"),
        "summary_txt": str(summary_txt),
    }
