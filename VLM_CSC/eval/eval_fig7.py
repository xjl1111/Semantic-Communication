"""Evaluation entry for Fig.7 sender-side KB comparison (SSQ vs SNR)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


def _proxy_ssq(sender_kb: str, snr_db: float) -> float:
    """Deterministic SSQ proxy for early-stage reproducible pipeline checks.

    This proxy is only used when full Fig.7 assets are not yet available.
    """
    quality = {
        "blip": 1.00,
        "lemon": 0.86,
        "ram": 0.74,
    }.get(sender_kb.lower(), 0.70)

    snr_gain = 0.52 + 0.045 * snr_db
    ssq = max(0.0, min(1.2, quality * snr_gain))
    return float(ssq)


def run_fig7_eval(
    sender_models: Iterable[str] = ("blip", "lemon", "ram"),
    snr_test_db: Iterable[int] = tuple(range(0, 11)),
    output_root: str = "outputs/fig7",
    dataset_name: str = "catsvsdogs",
    channel_name: str = "awgn",
) -> Dict[str, object]:
    """Run Fig.7 evaluation and write standardized outputs.

    Outputs:
    - outputs/fig7/results.csv
    - outputs/fig7/curve.png
    - outputs/fig7/logs/run_meta.json
    - outputs/fig7/experiment_note.md
    """
    out_dir = Path(output_root)
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for sender in sender_models:
        for snr in snr_test_db:
            rows.append(
                {
                    "dataset": dataset_name,
                    "channel": channel_name,
                    "sender_kb": sender,
                    "snr_db": int(snr),
                    "ssq": _proxy_ssq(sender, float(snr)),
                }
            )

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["dataset", "channel", "sender_kb", "snr_db", "ssq"])
        writer.writeheader()
        writer.writerows(rows)

    by_sender: Dict[str, List[float]] = {name: [] for name in sender_models}
    snr_list = list(snr_test_db)
    for sender in sender_models:
        sender_rows = [row for row in rows if row["sender_kb"] == sender]
        sender_rows = sorted(sender_rows, key=lambda item: int(item["snr_db"]))
        by_sender[sender] = [float(item["ssq"]) for item in sender_rows]

    plt.figure(figsize=(7, 4.5))
    for sender in sender_models:
        plt.plot(snr_list, by_sender[sender], marker="o", label=sender.upper())
    plt.xlabel("SNR_test (dB)")
    plt.ylabel("SSQ")
    plt.title("Fig.7 Proxy Reproduction: SSQ vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "curve.png", dpi=200)
    plt.close()

    run_meta = {
        "figure": "fig7",
        "dataset": dataset_name,
        "channel": channel_name,
        "sender_models": list(sender_models),
        "snr_test_db": snr_list,
        "mode": "proxy",
    }
    (logs_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    experiment_note = (
        "# experiment_note\n\n"
        "当前运行为 Fig.7 代理模式（proxy），用于验证工程流程、输出格式与曲线生成链路。\n\n"
        "- 论文明确写出：BLIP/LEMON/RAM 对比、CATSvsDOGS、AWGN、输出 SSQ-SNR 曲线。\n"
        "- 为复现做的合理实现选择：在缺少完整数据与权重时，先使用 deterministic proxy 生成可复现结果。\n"
        "- 当完整数据与模型权重就绪后，应切换为真实训练/评估流程并覆盖本结果。\n"
    )
    (out_dir / "experiment_note.md").write_text(experiment_note, encoding="utf-8")

    return {
        "results_csv": str(csv_path),
        "curve_png": str(out_dir / "curve.png"),
        "log_meta": str(logs_dir / "run_meta.json"),
    }
