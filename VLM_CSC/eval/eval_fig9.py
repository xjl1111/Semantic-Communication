"""Evaluation entry for Fig.9 NAM ablation (BLEU vs SNR_test)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


def _proxy_bleu_nam_on(snr_test: float) -> float:
    base = 0.42 + 0.038 * snr_test
    return float(max(0.05, min(0.92, base)))


def _proxy_bleu_nam_off(snr_train_fixed: float, snr_test: float) -> float:
    mismatch = abs(snr_test - snr_train_fixed)
    peak = 0.70 - 0.015 * mismatch
    floor = 0.20 + 0.02 * snr_test
    bleu = max(floor, peak)
    return float(max(0.04, min(0.88, bleu)))


def run_fig9_eval(
    snr_test_db: Iterable[int] = tuple(range(0, 11)),
    nam_off_train_snrs: Iterable[int] = (0, 2, 4, 8),
    output_root: str = "outputs/fig9",
) -> Dict[str, str]:
    """Run Fig.9 NAM ablation evaluation and generate CSV/curve/log artifacts."""
    out_dir = Path(output_root)
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    snr_test_values = list(snr_test_db)
    fixed_train = list(nam_off_train_snrs)

    rows: List[Dict[str, object]] = []
    for snr in snr_test_values:
        rows.append(
            {
                "model": "nam_on_uniform_0_10",
                "snr_train": "uniform(0,10)",
                "snr_test": int(snr),
                "bleu": _proxy_bleu_nam_on(float(snr)),
            }
        )
        for train_snr in fixed_train:
            rows.append(
                {
                    "model": f"nam_off_fixed_{train_snr}db",
                    "snr_train": float(train_snr),
                    "snr_test": int(snr),
                    "bleu": _proxy_bleu_nam_off(float(train_snr), float(snr)),
                }
            )

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "snr_train", "snr_test", "bleu"])
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(7.2, 4.6))
    nam_on_values = [
        float(r["bleu"]) for r in rows if r["model"] == "nam_on_uniform_0_10"
    ]
    plt.plot(snr_test_values, nam_on_values, marker="o", linewidth=2.2, label="NAM-ON train~U(0,10)")

    for train_snr in fixed_train:
        model_name = f"nam_off_fixed_{train_snr}db"
        values = [float(r["bleu"]) for r in rows if r["model"] == model_name]
        plt.plot(snr_test_values, values, marker=".", linestyle="--", label=f"NAM-OFF train={train_snr}dB")

    plt.xlabel("SNR_test (dB)")
    plt.ylabel("BLEU")
    plt.title("Fig.9 Proxy Reproduction: NAM Ablation")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    curve_path = out_dir / "curve.png"
    plt.savefig(curve_path, dpi=220)
    plt.close()

    run_meta = {
        "figure": "fig9",
        "mode": "proxy",
        "snr_test": snr_test_values,
        "nam_off_train_snrs": fixed_train,
    }
    (logs_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    note = (
        "# experiment_note\n\n"
        "当前运行为 Fig.9 代理模式（proxy），用于验证 NAM 消融曲线产物链路。\n\n"
        "- 论文明确写出：NAM-on 训练 SNR~Uniform(0,10)，NAM-off 在 0/2/4/8dB 固定训练，测试 0..10dB。\n"
        "- 为复现做的合理实现选择：在真实全训练前用 deterministic proxy 先打通 CSV 与曲线输出。\n"
        "- 完整实验时应替换为真实训练 BLEU 曲线。\n"
    )
    (out_dir / "experiment_note.md").write_text(note, encoding="utf-8")

    return {
        "results_csv": str(csv_path),
        "curve_png": str(curve_path),
        "log_meta": str(logs_dir / "run_meta.json"),
    }
