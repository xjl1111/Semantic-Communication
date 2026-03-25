"""Evaluation entry for Fig.8 MED ablation (continual BLEU map)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _proxy_bleu(task_train_idx: int, task_eval_idx: int, med_on: bool, order: List[str]) -> Tuple[float, float]:
    """Deterministic proxy BLEU for continual-learning map.

    - Without MED: stronger forgetting on older tasks after learning newer tasks.
    - With MED: forgetting is reduced.
    """
    if task_eval_idx > task_train_idx:
        return 0.0, 0.0

    distance = task_train_idx - task_eval_idx
    base = 0.58 + 0.05 * task_eval_idx
    forgetting = (0.20 if not med_on else 0.08) * distance
    bleu1 = max(0.02, min(0.95, base - forgetting))
    bleu2 = max(0.01, min(0.90, bleu1 - 0.06))
    return float(bleu1), float(bleu2)


def _save_map_png(matrix: np.ndarray, tasks: List[str], title: str, out_path: Path) -> None:
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(label="BLEU")
    plt.xticks(ticks=range(len(tasks)), labels=[t.upper() for t in tasks], rotation=15)
    plt.yticks(ticks=range(len(tasks)), labels=[t.upper() for t in tasks])
    plt.xlabel("Evaluate on task")
    plt.ylabel("After training task")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def run_fig8_eval(
    task_order: Iterable[str] = ("cifar", "birds", "catsvsdogs"),
    output_root: str = "outputs/fig8",
    channel_name: str = "rayleigh",
) -> Dict[str, str]:
    """Run Fig.8 MED ablation evaluation and generate standardized artifacts."""
    tasks = list(task_order)
    out_dir = Path(output_root)
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    matrices: Dict[str, np.ndarray] = {
        "med_off_bleu1": np.zeros((len(tasks), len(tasks)), dtype=float),
        "med_off_bleu2": np.zeros((len(tasks), len(tasks)), dtype=float),
        "med_on_bleu1": np.zeros((len(tasks), len(tasks)), dtype=float),
        "med_on_bleu2": np.zeros((len(tasks), len(tasks)), dtype=float),
    }

    for train_idx, train_task in enumerate(tasks):
        for eval_idx, eval_task in enumerate(tasks):
            b1_off, b2_off = _proxy_bleu(train_idx, eval_idx, med_on=False, order=tasks)
            b1_on, b2_on = _proxy_bleu(train_idx, eval_idx, med_on=True, order=tasks)

            matrices["med_off_bleu1"][train_idx, eval_idx] = b1_off
            matrices["med_off_bleu2"][train_idx, eval_idx] = b2_off
            matrices["med_on_bleu1"][train_idx, eval_idx] = b1_on
            matrices["med_on_bleu2"][train_idx, eval_idx] = b2_on

            rows.append(
                {
                    "channel": channel_name,
                    "train_task": train_task,
                    "eval_task": eval_task,
                    "med": "off",
                    "bleu1": b1_off,
                    "bleu2": b2_off,
                }
            )
            rows.append(
                {
                    "channel": channel_name,
                    "train_task": train_task,
                    "eval_task": eval_task,
                    "med": "on",
                    "bleu1": b1_on,
                    "bleu2": b2_on,
                }
            )

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["channel", "train_task", "eval_task", "med", "bleu1", "bleu2"])
        writer.writeheader()
        writer.writerows(rows)

    map_paths = {
        "med_off_bleu1": out_dir / "med_off_bleu1_map.png",
        "med_off_bleu2": out_dir / "med_off_bleu2_map.png",
        "med_on_bleu1": out_dir / "med_on_bleu1_map.png",
        "med_on_bleu2": out_dir / "med_on_bleu2_map.png",
    }
    _save_map_png(matrices["med_off_bleu1"], tasks, "Fig.8 Proxy: MED OFF BLEU-1", map_paths["med_off_bleu1"])
    _save_map_png(matrices["med_off_bleu2"], tasks, "Fig.8 Proxy: MED OFF BLEU-2", map_paths["med_off_bleu2"])
    _save_map_png(matrices["med_on_bleu1"], tasks, "Fig.8 Proxy: MED ON BLEU-1", map_paths["med_on_bleu1"])
    _save_map_png(matrices["med_on_bleu2"], tasks, "Fig.8 Proxy: MED ON BLEU-2", map_paths["med_on_bleu2"])

    run_meta = {
        "figure": "fig8",
        "mode": "proxy",
        "channel": channel_name,
        "task_order": tasks,
        "outputs": {name: str(path) for name, path in map_paths.items()},
    }
    (logs_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    note = (
        "# experiment_note\n\n"
        "当前运行为 Fig.8 代理模式（proxy），用于验证 continual learning map 产物链路。\n\n"
        "- 论文明确写出：任务顺序 CIFAR→BIRDS→CATSvsDOGS，Rayleigh 信道，比较有无 MED。\n"
        "- 为复现做的合理实现选择：在真实长期训练前先用 deterministic proxy 生成四张 BLEU map。\n"
        "- 完整实验时应替换为真实模型训练得到的 BLEU-1/2。\n"
    )
    (out_dir / "experiment_note.md").write_text(note, encoding="utf-8")

    return {
        "results_csv": str(csv_path),
        "med_off_bleu1_map": str(map_paths["med_off_bleu1"]),
        "med_off_bleu2_map": str(map_paths["med_off_bleu2"]),
        "med_on_bleu1_map": str(map_paths["med_on_bleu1"]),
        "med_on_bleu2_map": str(map_paths["med_on_bleu2"]),
        "log_meta": str(logs_dir / "run_meta.json"),
    }
