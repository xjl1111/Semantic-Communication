"""基线评估：JSCC/WITT/VLM-CSC 对比 + JSCC 管道构建。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from common import build_vlm_system, collect_binary_images_from_split as collect_eval_images
from eval.config import EvalConfig, load_module
from eval.validators import check_sd_assets
from eval.metrics import evaluate_perf_sender_snr


def _build_jscc_pipeline(project_root: Path, checkpoint: Path, channel_type: str, snr_db: float, device: str):
    import torch
    from torchvision import transforms

    deep_jscc_root = project_root / "deep_jscc"
    encoder_module = load_module("deep_jscc_encoder_module", deep_jscc_root / "model" / "encoder.py")
    decoder_module = load_module("deep_jscc_decoder_module", deep_jscc_root / "model" / "decoder.py")
    channel_module = load_module("deep_jscc_channel_module", deep_jscc_root / "model" / "channel.py")
    Encoder = getattr(encoder_module, "Encoder")
    Decoder = getattr(decoder_module, "Decoder")
    Channel = getattr(channel_module, "Channel")

    encoder = Encoder(in_channels=3, c=128).to(device).eval()
    decoder = Decoder(in_channels=128, out_channels=3).to(device).eval()
    channel = Channel(channel_type=channel_type, snr_db=snr_db)

    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict):
        if "encoder" in state:
            encoder.load_state_dict(state["encoder"], strict=True)
        if "decoder" in state:
            decoder.load_state_dict(state["decoder"], strict=True)

    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    return encoder, decoder, channel, preprocess


def _run_baseline_performance(
    config: EvalConfig,
    target_module,
) -> Dict[str, str]:
    import csv
    import torch

    # ⚠ JSCC / WITT 基线阻断项 (BLOCKER)
    # 论文 Fig.10 要求对比 JSCC 和 WITT 基线，但：
    #   1. 论文未提供 JSCC/WITT 的精确实现代码或训练好的权重
    #   2. deep_jscc/ 目录下的实现为本项目自行编写，无法确认与论文一致
    #   3. WITT 使用 torch.jit.load 加载 TorchScript 模型，需要原作者提供
    #   4. 论文未精确定义 compression_ratio 的计算方式
    # 因此：JSCC/WITT 基线为严格复现阻断项，仅 vlm_csc 基线可正式运行
    _blocked_baselines = {"jscc", "witt"}
    requested_baselines = set(str(b).lower() for b in (config.baselines or []))
    blocked = requested_baselines & _blocked_baselines
    if blocked:
        raise RuntimeError(
            f"[BLOCKER] Fig.10 基线 {sorted(blocked)} 为严格复现阻断项：\n"
            f"  - 论文未提供 JSCC/WITT 的公开实现或预训练权重\n"
            f"  - 论文未精确定义 compression_ratio 的计算方式\n"
            f"  - 不得使用未经验证的第三方实现作为正式基线\n"
            f"如需仅评估 vlm_csc，请将 baselines 设为 ['vlm_csc']。"
        )

    project_root = Path(config.project_root)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_eval_images(Path(config.test_split_dir), max_per_class=config.max_per_class)
    if len(records) == 0:
        raise RuntimeError("No evaluation images found for baseline evaluation")

    if config.baseline_checkpoints is None:
        raise RuntimeError("Fig10 baseline mode requires baseline_checkpoints mapping")

    classifier = target_module.CatsDogsDownstreamClassifier(device="cuda")
    rows = []
    align_dir = output_dir / f"fig_{config.tag}_alignment_examples"
    if config.export_alignment_examples:
        align_dir.mkdir(parents=True, exist_ok=True)

    vlm_model = None
    vlm_module = None
    vlm_sender = ""
    if any(str(x).lower() == "vlm_csc" for x in (config.baselines or [])):
        if not config.senders:
            raise RuntimeError("Fig10 baseline mode requires at least one sender for vlm_csc baseline.")
        vlm_sender = str(config.senders[0]).lower()
        blip_dir = Path(config.blip_ckb_dir)
        ram_ckpt = Path(config.ram_ckb_path)
        sd_dir = Path(config.sd_ckb_dir)
        if not blip_dir.exists():
            raise RuntimeError(f"BLIP CKB directory not found: {blip_dir}")
        if not ram_ckpt.exists():
            raise RuntimeError(f"RAM CKB weight not found: {ram_ckpt}")
        if not sd_dir.exists():
            raise RuntimeError(f"SD CKB directory not found: {sd_dir}")
        missing_sd = check_sd_assets(sd_dir)
        if missing_sd:
            raise RuntimeError(f"SD local assets incomplete under hard protocol mode: {missing_sd}")

        vlm_module = load_module("vlm_csc_module_fig10_baseline", Path(config.model_file))

    for baseline in (config.baselines or []):
        baseline = str(baseline).lower()
        ckpt_path = Path(config.baseline_checkpoints.get(baseline, "")).expanduser().resolve()
        if not ckpt_path.exists():
            raise RuntimeError(f"Baseline checkpoint missing for {baseline}: {ckpt_path}")

        if baseline == "vlm_csc":
            if vlm_model is None:
                vlm_model = build_vlm_system(
                    vlm_module,
                    sender=vlm_sender,
                    blip_dir=Path(config.blip_ckb_dir),
                    ram_ckpt=Path(config.ram_ckb_path),
                    sd_dir=Path(config.sd_ckb_dir),
                    channel_type=config.channel_type,
                    device="cuda",
                    quiet_third_party=config.quiet_third_party,
                    enable_med=False,
                    med_kwargs=None,
                    max_text_len=int(config.max_text_len),
                    max_text_len_by_sender=config.max_text_len_by_sender,
                    channel_dim=config.channel_dim,
                    caption_mode=config.caption_mode,
                    caption_prompt=config.caption_prompt,
                )
                state = torch.load(ckpt_path, map_location="cuda")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                vlm_model.load_state_dict(state, strict=True)
                vlm_model.eval()

        for snr_db in config.snr_list:
            snr_db = float(snr_db)
            labels: List[int] = []
            preds: List[int] = []
            src_lengths: List[int] = []
            tx_lengths: List[int] = []

            if baseline == "jscc":
                encoder, decoder, channel, preprocess = _build_jscc_pipeline(project_root, ckpt_path, config.channel_type, snr_db, "cuda")

                for idx, rec in enumerate(records):
                    original = Image.open(rec["path"]).convert("RGB")
                    x = preprocess(original).unsqueeze(0).to("cuda")  # type: ignore[union-attr]
                    with torch.no_grad():
                        z = encoder(x)
                        y = channel(z, snr_db=snr_db)
                        x_hat = decoder(y).clamp(0, 1)
                    rec_img = Image.fromarray((x_hat[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)).resize(original.size)
                    preds.append(classifier.predict_label(rec_img))
                    labels.append(int(rec["label"]))
                    src_lengths.append(int(np.prod(x.shape[1:])))
                    tx_lengths.append(int(np.prod(z.shape[1:])))

                    if config.export_alignment_examples and idx < 3:
                        rec_img.save(align_dir / f"{baseline}_snr{int(snr_db)}_{idx}.png")

            elif baseline == "witt":
                model = torch.jit.load(str(ckpt_path), map_location="cuda")
                model.eval()
                for idx, rec in enumerate(records):
                    original = Image.open(rec["path"]).convert("RGB")
                    arr = np.asarray(original.resize((256, 256)), dtype=np.float32) / 255.0
                    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to("cuda")
                    with torch.no_grad():
                        x_hat = model(x, torch.tensor([snr_db], device="cuda"))
                    rec_img = Image.fromarray((x_hat[0].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)).resize(original.size)
                    preds.append(classifier.predict_label(rec_img))
                    labels.append(int(rec["label"]))
                    src_lengths.append(int(np.prod(x.shape[1:])))
                    tx_lengths.append(int(np.prod(x.shape[1:])))
                    if config.export_alignment_examples and idx < 3:
                        rec_img.save(align_dir / f"{baseline}_snr{int(snr_db)}_{idx}.png")

            elif baseline == "vlm_csc":
                perf_scores, _, _ = evaluate_perf_sender_snr(
                    model=vlm_model,
                    sender_type=baseline,
                    snr_db=snr_db,
                    records=records,
                    target_module=target_module,
                    sd_steps=config.sd_steps,
                    sd_height=config.sd_height,
                    sd_width=config.sd_width,
                    batch_size=config.batch_size,
                    max_batches=config.max_batches,
                    device="cuda",
                    seed=config.seed,
                )
                rows.append(
                    {
                        "sender": baseline,
                        "snr_db": snr_db,
                        "metric": "classification_accuracy",
                        "value": float(perf_scores["classification_accuracy"]),
                    }
                )
                rows.append(
                    {
                        "sender": baseline,
                        "snr_db": snr_db,
                        "metric": "compression_ratio",
                        "value": float(perf_scores["compression_ratio"]),
                    }
                )
                rows.append(
                    {
                        "sender": baseline,
                        "snr_db": snr_db,
                        "metric": "trainable_parameters",
                        "value": float(perf_scores["trainable_parameters"]),
                    }
                )
                continue
            else:
                raise RuntimeError(f"Unsupported baseline: {baseline}")

            rows.append(
                {
                    "sender": baseline,
                    "snr_db": snr_db,
                    "metric": "classification_accuracy",
                    "value": float(target_module.compute_classification_accuracy(preds, labels)),
                }
            )
            rows.append(
                {
                    "sender": baseline,
                    "snr_db": snr_db,
                    "metric": "compression_ratio",
                    "value": float(target_module.compute_compression_ratio(src_lengths, tx_lengths)),
                }
            )

    curve_csv = output_dir / f"fig_{config.tag}_baselines_curve.csv"
    with curve_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sender", "snr_db", "metric", "value"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_txt = output_dir / f"fig_{config.tag}_baselines_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write(f"baselines: {config.baselines}\n")
        f.write(f"baseline_checkpoints: {config.baseline_checkpoints}\n")

    return {
        "detail_csv": str(curve_csv),
        "curve_csv": str(curve_csv),
        "plot_png": "",
        "summary_txt": str(summary_txt),
    }
