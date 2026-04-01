from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

# ─── 文本语义层级 (Level-A / Level-B) 用到的 cat/dog 词汇正则 ───────────────────
_CAT_PATTERN = re.compile(r"\b(cat|cats|kitten|kittens|kitty|feline)\b", re.IGNORECASE)
_DOG_PATTERN = re.compile(r"\b(dog|dogs|puppy|puppies|pup|canine|hound)\b", re.IGNORECASE)


def _text_matches_label(text: str, label: int) -> bool:
    """文本是否《排他性》匹配整数标签对应的类别词（0=cat, 1=dog）。

    正确内容：目标动物出现 **且** 错误动物未出现。
    若同时出现两种动物（歧义 caption），判为错误。
    """
    has_cat = bool(_CAT_PATTERN.search(text))
    has_dog = bool(_DOG_PATTERN.search(text))
    if label == 0:
        return has_cat and not has_dog
    if label == 1:
        return has_dog and not has_cat
    return False

from common import chunk_records


def maybe_black_image(image: Image.Image) -> bool:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return bool(arr.mean() < 1.0)


def evaluate_ssq_sender_snr(
    model,
    sender_type: str,
    snr_db: float,
    records,
    target_module,
    sd_steps: int,
    sd_height: int = 512,
    sd_width: int = 512,
    batch_size: int = 32,
    max_batches: int = -1,
    device: str = "cuda",
    seed: int = 42,
    collect_visual_samples: int = 10,
    clip_classifier_path: str = "",
):
    classifier = target_module.CatsDogsDownstreamClassifier(
        device=device, finetuned_clip_path=clip_classifier_path,
    )
    if len(classifier.degradation_notes) > 0:
        raise RuntimeError(
            f"Downstream classifier entered degradation path for sender={sender_type}, snr={snr_db}: "
            f"{classifier.degradation_notes}"
        )

    preds_original: List[int] = []
    preds_reconstructed: List[int] = []
    labels: List[int] = []
    detail_rows = []
    black_output_count = 0
    visual_samples: List[Dict] = []  # 前 collect_visual_samples 张的可视化数据
    # Level-A/B 计数器（仅在 cat/dog 数据集上有效，label ∈ {0, 1}）
    text_src_correct = 0
    text_rec_correct = 0
    text_ab_total = 0

    batches = chunk_records(records, batch_size=batch_size, max_batches=max_batches)
    sample_index = 0
    for batch in tqdm(batches, desc=f"eval/test ssq ({sender_type}, snr={snr_db})", leave=False):
        for rec in batch:
            image_path = rec["path"]
            label = int(rec["label"])
            original = Image.open(image_path).convert("RGB")

            out = model.infer_full(
                image=original,
                snr_db=snr_db,
                sd_height=sd_height,
                sd_width=sd_width,
                sd_steps=sd_steps,
                sd_guidance=7.5,
                sd_seed=seed + sample_index,
                return_debug=False,
            )

            reconstructed = out["reconstructed_image"].convert("RGB").resize(original.size)
            if maybe_black_image(reconstructed):
                black_output_count += 1

            pred_o = classifier.predict_label(original)
            pred_r = classifier.predict_label(reconstructed)

            preds_original.append(pred_o)
            preds_reconstructed.append(pred_r)
            labels.append(label)

            detail_rows.append(
                {
                    "sender": sender_type,
                    "snr_db": snr_db,
                    "image": image_path.name,
                    "label": label,
                    "pred_original": pred_o,
                    "pred_reconstructed": pred_r,
                    "source_text": out["source_text"],
                    "recovered_text": out["recovered_text"],
                }
            )

            # Level-A/B（仅对 cat/dog 数据有效）
            if label in (0, 1):
                src_txt = str(out["source_text"] or "")
                rec_txt = str(out["recovered_text"] or "")
                if _text_matches_label(src_txt, label):
                    text_src_correct += 1
                if _text_matches_label(rec_txt, label):
                    text_rec_correct += 1
                text_ab_total += 1

            # 收集可视化样本
            if len(visual_samples) < collect_visual_samples:
                visual_samples.append({
                    "original": original.copy(),
                    "reconstructed": reconstructed.copy(),
                    "source_text": str(out["source_text"]),
                    "recovered_text": str(out["recovered_text"]),
                    "label": label,
                    "pred_original": pred_o,
                    "pred_reconstructed": pred_r,
                    "image_name": image_path.name,
                })

            sample_index += 1

    ssq_res = target_module.compute_ssq(preds_original, preds_reconstructed, labels)
    if black_output_count > 0:
        print(f"[WARN] possible NSFW/black output count: sender={sender_type}, snr={snr_db}, count={black_output_count}")
    # ── ABCD 四维监控摘要 ──────────────────────────────────────────────────────
    if text_ab_total > 0:
        level_a = text_src_correct / text_ab_total
        level_b = text_rec_correct / text_ab_total
        print(
            f"[eval/ABCD] sender={sender_type}  snr={snr_db:+.0f}dB  "
            f"A(src_txt)={level_a:.1%}  B(rec_txt)={level_b:.1%}  "
            f"C(orig_clf)={ssq_res.st_original:.1%}  D(recon_clf)={ssq_res.st_reconstructed:.1%}  "
            f"SSQ={ssq_res.ssq:.3f}"
        )
    return ssq_res, detail_rows, classifier.backend, visual_samples


def evaluate_bleu_sender_snr(
    model,
    sender_type: str,
    snr_db: float,
    records,
    target_module,
    sd_steps: int,
    sd_height: int = 512,
    sd_width: int = 512,
    batch_size: int = 32,
    max_batches: int = -1,
    seed: int = 42,
):
    references: List[str] = []
    hypotheses: List[str] = []
    detail_rows = []

    batches = chunk_records(records, batch_size=batch_size, max_batches=max_batches)
    sample_index = 0
    for batch in tqdm(batches, desc=f"eval/test bleu ({sender_type}, snr={snr_db})", leave=False):
        for rec in batch:
            image_path = rec["path"]
            original = Image.open(image_path).convert("RGB")
            out = model.infer_full(
                image=original,
                snr_db=snr_db,
                sd_height=sd_height,
                sd_width=sd_width,
                sd_steps=sd_steps,
                sd_guidance=7.5,
                sd_seed=seed + sample_index,
                return_debug=False,
            )
            references.append(str(out["source_text"]))
            hypotheses.append(str(out["recovered_text"]))
            detail_rows.append(
                {
                    "sender": sender_type,
                    "snr_db": snr_db,
                    "image": image_path.name,
                    "source_text": out["source_text"],
                    "recovered_text": out["recovered_text"],
                }
            )
            sample_index += 1

    bleu1 = float(target_module.compute_bleu1(references, hypotheses))
    bleu2 = float(target_module.compute_bleu2(references, hypotheses))
    return {"bleu": bleu1, "bleu1": bleu1, "bleu2": bleu2}, detail_rows


def evaluate_perf_sender_snr(
    model,
    sender_type: str,
    snr_db: float,
    records,
    target_module,
    sd_steps: int,
    sd_height: int = 512,
    sd_width: int = 512,
    batch_size: int = 32,
    max_batches: int = -1,
    device: str = "cuda",
    seed: int = 42,
):
    classifier = target_module.CatsDogsDownstreamClassifier(device=device)
    labels: List[int] = []
    preds_reconstructed: List[int] = []
    source_lengths: List[int] = []
    transmitted_lengths: List[int] = []
    detail_rows = []

    batches = chunk_records(records, batch_size=batch_size, max_batches=max_batches)
    sample_index = 0
    for batch in tqdm(batches, desc=f"eval/test perf ({sender_type}, snr={snr_db})", leave=False):
        for rec in batch:
            image_path = rec["path"]
            label = int(rec["label"])
            original = Image.open(image_path).convert("RGB")
            out = model.infer_full(
                image=original,
                snr_db=snr_db,
                sd_height=sd_height,
                sd_width=sd_width,
                sd_steps=sd_steps,
                sd_guidance=7.5,
                sd_seed=seed + sample_index,
                return_debug=True,
            )

            reconstructed = out["reconstructed_image"].convert("RGB").resize(original.size)
            pred_r = classifier.predict_label(reconstructed)
            labels.append(label)
            preds_reconstructed.append(pred_r)

            src_len = int((out["source_token_ids"][0] != model.tokenizer.pad_id).sum().item())
            tx_len = int((out["token_ids"][0] != model.tokenizer.pad_id).sum().item())
            source_lengths.append(src_len)
            transmitted_lengths.append(max(tx_len, 1))

            detail_rows.append(
                {
                    "sender": sender_type,
                    "snr_db": snr_db,
                    "image": image_path.name,
                    "label": label,
                    "pred_reconstructed": pred_r,
                    "source_text": out["source_text"],
                    "recovered_text": out["recovered_text"],
                }
            )
            sample_index += 1

    acc = float(target_module.compute_classification_accuracy(preds_reconstructed, labels))
    compression = float(target_module.compute_compression_ratio(source_lengths, transmitted_lengths))
    params = float(target_module.count_trainable_parameters(model))
    return {
        "classification_accuracy": acc,
        "compression_ratio": compression,
        "trainable_parameters": params,
    }, detail_rows, classifier.backend


# ══════════════════════════════════════════════════════════════════════════════
# 可视化：每组 SNR 的样本对比大图
# ══════════════════════════════════════════════════════════════════════════════

import textwrap as _tw


def save_visual_samples(
    visual_samples: List[Dict],
    sender: str,
    snr_db: float,
    out_dir: Path,
    ssq_value: float = 0.0,
    st_original: float = 0.0,
    st_reconstructed: float = 0.0,
) -> Path:
    """为一个 sender×SNR 生成可视化大图。

    布局：N 列, 3 行
      Row 0 : 原图
      Row 1 : 重构图
      Row 2 : 文本对比 (source_text | recovered_text | 分类结果)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from matplotlib.gridspec import GridSpec as _GS

    n = len(visual_samples)
    if n == 0:
        return out_dir / "empty.png"

    LABEL_MAP = {0: "cat", 1: "dog"}
    col_w = 3.0
    fig_w = max(n * col_w, 8)
    fig_h = 10.0

    fig = _plt.figure(figsize=(fig_w, fig_h))
    gs = _GS(3, n, figure=fig, height_ratios=[2, 2, 2.5], hspace=0.3, wspace=0.08)

    title = (
        f"{sender.upper()}  |  SNR = {snr_db:.0f} dB  |  "
        f"SSQ = {ssq_value:.4f}  "
        f"(ST_orig={st_original:.4f}, ST_recon={st_reconstructed:.4f})"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    for col, s in enumerate(visual_samples):
        gt_label = LABEL_MAP.get(s["label"], str(s["label"]))
        pred_o_label = LABEL_MAP.get(s["pred_original"], str(s["pred_original"]))
        pred_r_label = LABEL_MAP.get(s["pred_reconstructed"], str(s["pred_reconstructed"]))

        # 分类是否正确
        o_correct = (s["pred_original"] == s["label"])
        r_correct = (s["pred_reconstructed"] == s["label"])
        o_color = "#27ae60" if o_correct else "#e74c3c"
        r_color = "#27ae60" if r_correct else "#e74c3c"

        # Row 0: 原图
        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(s["original"])
        ax0.set_title(
            f"GT={gt_label}  CLIP={pred_o_label}",
            fontsize=10, color=o_color, fontweight="bold", pad=3,
        )
        ax0.axis("off")

        # Row 1: 重构图
        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(s["reconstructed"])
        ax1.set_title(
            f"CLIP={pred_r_label}  {'OK' if r_correct else 'WRONG'}",
            fontsize=10, color=r_color, fontweight="bold", pad=3,
        )
        ax1.axis("off")

        # Row 2: 文本对比
        ax2 = fig.add_subplot(gs[2, col])
        ax2.axis("off")

        src_w = "\n".join(_tw.wrap(str(s["source_text"]),    width=28))
        rec_w = "\n".join(_tw.wrap(str(s["recovered_text"]), width=28))
        same_text = (str(s["source_text"]).strip() == str(s["recovered_text"]).strip())
        text_match = "[MATCH]" if same_text else "[DIFF]"
        text_match_color = "#27ae60" if same_text else "#e74c3c"

        body = (
            f"Source:\n{src_w}\n\n"
            f"Recovered:\n{rec_w}\n\n"
            f"{text_match}"
        )
        ax2.text(
            0.5, 0.97, body,
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="center",
            multialignment="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                      edgecolor=text_match_color, linewidth=1.5, alpha=0.9),
        )

    # 行标签
    for row, lbl in enumerate(["Original", "Reconstructed", "Captions"]):
        ax = fig.add_subplot(gs[row, 0])
        ax.set_ylabel(lbl, fontsize=11, rotation=90, labelpad=6, va="center", color="#333")

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"visual_{sender}_snr_{snr_db:+.0f}db.png"
    out_path = out_dir / fname
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    _plt.close(fig)
    return out_path


def plot_curve(curve_rows: List[Dict], fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    senders = sorted(set(row["sender"] for row in curve_rows))

    plt.figure(figsize=(8, 5))
    for sender in senders:
        rows = [r for r in curve_rows if r["sender"] == sender]
        rows = sorted(rows, key=lambda x: x["snr_db"])
        x = [r["snr_db"] for r in rows]
        y = [r["ssq"] for r in rows]
        plt.plot(x, y, marker="o", label=sender.upper())

    plt.title("Fig: SSQ vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SSQ = ST(reconstructed)/ST(original)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def plot_matrix_heatmap(matrix: np.ndarray, row_labels: List[str], col_labels: List[str], out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.imshow(matrix, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(col_labels)), col_labels, rotation=30, ha="right")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
