"""评估核心：通用 SNR 循环 + CSV/plot 输出。"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from common import (
    build_vlm_system,
    chunk_records,
    collect_binary_images_from_split as collect_eval_images,
    configure_runtime_logging,
)
from eval.config import EvalConfig, load_module
from eval.validators import (
    _file_sha256,
    _validate_fig8_strict_protocol_inputs,
    check_sd_assets,
)
from eval.fig8_continual import _run_continual_bleu_map
from eval.baselines import _run_baseline_performance
from eval.metrics import (
    evaluate_bleu_sender_snr,
    evaluate_perf_sender_snr,
    evaluate_ssq_sender_snr,
    plot_curve,
    save_visual_samples,
)


def _run_evaluation_core(config: EvalConfig) -> Dict[str, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment, but no CUDA device is available.")
    fig_name = str(config.fig_name).strip().lower()
    min_sd_steps = 1 if fig_name in {"fig8", "fig9"} else 2
    if int(config.sd_steps) < min_sd_steps:
        if min_sd_steps == 1:
            raise RuntimeError(f"sd_steps must be >= 1 for {fig_name} under hard protocol mode.")
        raise RuntimeError("sd_steps must be > 1 under hard protocol mode.")
    if not config.strict_ckpt:
        raise RuntimeError("strict_ckpt must be enabled under hard protocol mode.")
    if config.strict_paper_repro and config.channel_type not in {"awgn", "rayleigh"}:
        raise RuntimeError(f"Invalid channel_type under strict mode: {config.channel_type}")

    in_fig8_mode = config.fig_name == "fig8" and config.eval_output_mode == "continual_learning_map"
    if not in_fig8_mode and len(config.snr_list) == 0:
        raise RuntimeError("snr_list is empty.")

    metrics = config.metrics if config.metrics is not None else ["ssq"]
    metrics = [str(m).lower() for m in metrics]
    allowed_metrics = {
        "ssq",
        "bleu",
        "bleu1",
        "bleu2",
        "classification_accuracy",
        "compression_ratio",
        "trainable_parameters",
    }
    for metric in metrics:
        if metric not in allowed_metrics:
            raise RuntimeError(f"Unsupported metric: {metric}")

    if config.fig_name == "fig8" and config.strict_paper_repro:
        _validate_fig8_strict_protocol_inputs(config, metrics)

    configure_runtime_logging(config.quiet_third_party)

    project_root = Path(config.project_root)
    model_file = Path(config.model_file)
    target_file = Path(config.target_file)
    test_split_dir = Path(config.test_split_dir) if str(config.test_split_dir).strip() != "" else None
    output_dir = Path(config.output_dir)

    blip_dir = Path(config.blip_ckb_dir)
    ram_ckpt = Path(config.ram_ckb_path) if config.ram_ckb_path else None
    sd_dir = Path(config.sd_ckb_dir)
    ckpt_blip_path = Path(config.ckpt_blip).expanduser().resolve()
    ckpt_ram_path = Path(config.ckpt_ram).expanduser().resolve() if config.ckpt_ram else None
    need_ram = "ram" in config.senders

    in_continual_mode = (
        bool(config.dataset_sequence)
        and any(m in metrics for m in ["bleu", "bleu1", "bleu2"])
        and config.eval_output_mode == "continual_learning_map"
    )
    if not in_continual_mode:
        if not ckpt_blip_path.exists():
            raise RuntimeError(f"BLIP checkpoint not found: {ckpt_blip_path}")
        if need_ram and (ckpt_ram_path is None or not ckpt_ram_path.exists()):
            raise RuntimeError(f"RAM checkpoint not found: {ckpt_ram_path}")
    if not blip_dir.exists():
        raise RuntimeError(f"BLIP CKB directory not found: {blip_dir}")
    if need_ram and (ram_ckpt is None or not ram_ckpt.exists()):
        raise RuntimeError(f"RAM CKB weight not found: {ram_ckpt}")
    if not sd_dir.exists():
        raise RuntimeError(f"SD CKB directory not found: {sd_dir}")

    missing_sd = check_sd_assets(sd_dir)
    if missing_sd:
        raise RuntimeError(f"SD local assets incomplete under hard protocol mode: {missing_sd}")

    vlm_module = load_module("vlm_csc_module", model_file)
    target_module = load_module("target_module", target_file)

    if in_continual_mode:
        return _run_continual_bleu_map(
            config=config,
            vlm_module=vlm_module,
            target_module=target_module,
            blip_dir=blip_dir,
            ram_ckpt=ram_ckpt,
            sd_dir=sd_dir,
        )

    if config.baselines and any(m in metrics for m in ["classification_accuracy", "compression_ratio", "trainable_parameters"]):
        baseline_out = _run_baseline_performance(config=config, target_module=target_module)
        return baseline_out

    if test_split_dir is None:
        raise RuntimeError("Non-continual evaluation requires test_split_dir to be configured.")
    records = collect_eval_images(test_split_dir, max_per_class=config.max_per_class)
    if len(records) == 0:
        raise RuntimeError("No evaluation images found in test split.")
    selected_batches = len(chunk_records(records, batch_size=config.batch_size, max_batches=config.max_batches))

    curve_rows = []
    detail_rows = []
    device = "cuda"
    backends_seen = set()
    sender_backend = {}

    for sender in tqdm(config.senders, desc="sender loop"):
        model = build_vlm_system(
            vlm_module,
            sender=sender,
            blip_dir=blip_dir,
            ram_ckpt=ram_ckpt,
            sd_dir=sd_dir,
            channel_type=config.channel_type,
            device=device,
            quiet_third_party=config.quiet_third_party,
            enable_med=False,
            med_kwargs=None,
            max_text_len=int(config.max_text_len),
            max_text_len_by_sender=config.max_text_len_by_sender,
            use_nam=bool(config.use_nam),
            channel_dim=config.channel_dim,
            caption_mode=config.caption_mode,
            caption_prompt=config.caption_prompt,
        )

        # --- Structural NAM assertion (anti-cheat) ---
        nam_param_count = sum(1 for n, _ in model.named_parameters() if "nam" in n.lower())
        if config.use_nam and nam_param_count == 0:
            raise RuntimeError(f"use_nam=True but model has 0 NAM params — structural integrity violation (sender={sender})")
        if (not config.use_nam) and nam_param_count > 0:
            raise RuntimeError(
                f"use_nam=False but model still has {nam_param_count} NAM params — "
                f"structural removal failed (sender={sender})"
            )

        ckpt_path = ckpt_blip_path if sender == "blip" else ckpt_ram_path
        raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # ── caption_mode 一致性校验（训练/评估必须相同）──────────────────────
        if isinstance(raw_ckpt, dict) and "caption_mode" in raw_ckpt:
            ckpt_mode = raw_ckpt["caption_mode"]
            cfg_mode  = config.caption_mode
            if ckpt_mode != cfg_mode:
                raise RuntimeError(
                    f"[caption_mode 不匹配] checkpoint 训练时 caption_mode='{ckpt_mode}'，"
                    f"当前评估配置 caption_mode='{cfg_mode}'。\n"
                    f"请将 CAPTION_MODE 改回 '{ckpt_mode}'，或用 '{cfg_mode}' 重新训练。"
                )
        state = raw_ckpt
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        model.eval()

        for snr_db in config.snr_list:
            snr_db = float(snr_db)
            if "ssq" in metrics:
                ssq_res, rows, backend, visual_samples = evaluate_ssq_sender_snr(
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
                    device=device,
                    seed=config.seed,
                    collect_visual_samples=10,
                    clip_classifier_path=getattr(config, "clip_classifier_path", ""),
                )
                detail_rows.extend(rows)
                curve_rows.append(
                    {
                        "sender": sender,
                        "snr_db": snr_db,
                        "metric": "ssq",
                        "value": ssq_res.ssq,
                        "st_original": ssq_res.st_original,
                        "st_reconstructed": ssq_res.st_reconstructed,
                        "classifier_backend": backend,
                    }
                )
                if config.required_classifier_backend and backend != config.required_classifier_backend:
                    raise RuntimeError(
                        f"Classifier backend mismatch: expected={config.required_classifier_backend}, got={backend}, "
                        f"sender={sender}, snr={snr_db}"
                    )
                backends_seen.add(backend)
                sender_backend[sender] = backend
                print(f"[eval][{sender}] SNR={snr_db:+.1f}dB  ssq={ssq_res.ssq:.4f}  "
                      f"st_orig={ssq_res.st_original:.4f}  st_recon={ssq_res.st_reconstructed:.4f}")

                # ── 保存可视化样本对比图 ───────────────────────────
                if visual_samples:
                    vis_dir = output_dir / "visual_samples"
                    vis_path = save_visual_samples(
                        visual_samples=visual_samples,
                        sender=sender,
                        snr_db=snr_db,
                        out_dir=vis_dir,
                        ssq_value=ssq_res.ssq,
                        st_original=ssq_res.st_original,
                        st_reconstructed=ssq_res.st_reconstructed,
                    )
                    print(f"         visual: {vis_path.name}")

            elif any(m in metrics for m in ["bleu", "bleu1", "bleu2"]):
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
                detail_rows.extend(rows)
                for metric_name in metrics:
                    if metric_name in bleu_scores:
                        curve_rows.append(
                            {
                                "sender": sender,
                                "snr_db": snr_db,
                                "metric": metric_name,
                                "value": float(bleu_scores[metric_name]),
                                "classifier_backend": "n/a",
                            }
                        )
                sender_backend[sender] = "n/a"
                bleu1 = float(bleu_scores.get("bleu1", bleu_scores.get("bleu", 0.0)))
                bleu2 = float(bleu_scores.get("bleu2", 0.0))
                print(f"[eval][{sender}] SNR={snr_db:+.1f}dB  bleu1={bleu1:.4f}  bleu2={bleu2:.4f}")

            else:
                perf_scores, rows, backend = evaluate_perf_sender_snr(
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
                    device=device,
                    seed=config.seed,
                )
                detail_rows.extend(rows)
                for metric_name in metrics:
                    curve_rows.append(
                        {
                            "sender": sender,
                            "snr_db": snr_db,
                            "metric": metric_name,
                            "value": float(perf_scores[metric_name]),
                            "classifier_backend": backend,
                        }
                    )
                if config.required_classifier_backend and backend != config.required_classifier_backend:
                    raise RuntimeError(
                        f"Classifier backend mismatch: expected={config.required_classifier_backend}, got={backend}, "
                        f"sender={sender}, snr={snr_db}"
                    )
                backends_seen.add(backend)
                sender_backend[sender] = backend
                perf_str = "  ".join(f"{m}={float(perf_scores[m]):.4f}" for m in metrics if m in perf_scores)
                print(f"[eval][{sender}] SNR={snr_db:+.1f}dB  {perf_str}")

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = output_dir / f"fig_{config.tag}_details.csv"
    curve_csv = output_dir / f"fig_{config.tag}_curve.csv"
    plot_png = output_dir / f"fig_{config.tag}_plot.png"
    summary_txt = output_dir / f"fig_{config.tag}_summary.txt"

    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sender",
                "snr_db",
                "image",
                "label",
                "pred_original",
                "pred_reconstructed",
                "source_text",
                "recovered_text",
            ],
        )
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)

    with curve_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sender", "snr_db", "metric", "value", "st_original", "st_reconstructed", "classifier_backend"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in curve_rows:
            writer.writerow(row)

    with summary_txt.open("w", encoding="utf-8") as f:
        f.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"project_root: {project_root}\n")
        f.write(f"test_split_dir: {test_split_dir}\n")
        f.write(f"snr_list: {config.snr_list}\n")
        f.write(f"batch_size: {config.batch_size}\n")
        f.write(f"max_batches: {config.max_batches}\n")
        f.write(f"max_per_class: {config.max_per_class}\n")
        f.write(f"selected_images: {len(records)}\n")
        f.write(f"selected_batches: {selected_batches}\n")
        f.write(f"required_classifier_backend: {config.required_classifier_backend}\n")
        f.write(f"classifier_backends_seen: {sorted(backends_seen)}\n")
        f.write(f"sender_backend: {sender_backend}\n")
        f.write(f"ckpt_blip: {ckpt_blip_path}\n")
        f.write(f"ckpt_blip_sha256: {_file_sha256(ckpt_blip_path)}\n")
        if need_ram and ckpt_ram_path is not None:
            f.write(f"ckpt_ram: {ckpt_ram_path}\n")
            f.write(f"ckpt_ram_sha256: {_file_sha256(ckpt_ram_path)}\n")

    if "ssq" in metrics:
        plot_curve(
            curve_rows=[
                {"sender": r["sender"], "snr_db": r["snr_db"], "ssq": r["value"]}
                for r in curve_rows
                if r.get("metric") == "ssq"
            ],
            fig_path=plot_png,
        )
    return {
        "detail_csv": str(detail_csv),
        "curve_csv": str(curve_csv),
        "plot_png": str(plot_png),
        "summary_txt": str(summary_txt),
    }
