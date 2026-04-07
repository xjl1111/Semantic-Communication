"""
Generate Fig.7 SSQ comparison plots.

Usage:
    python plot_blip_results.py                 # 绘制所有已有结果
    python plot_blip_results.py --run blip_ft   # 仅绘制单个 run
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parents[1] / "data" / "experiments" / "fig7" / "runs"

# ── 颜色 / 样式注册表 ──
STYLE = {
    "blip_ft_prompt":   {"color": "#2563eb", "marker": "o",  "label": "BLIP (finetuned)"},
    "blip_noFt_prompt": {"color": "#dc2626", "marker": "s",  "label": "BLIP (no finetune)"},
}


def _read_run(run_dir: Path):
    """从 run 目录中读取第一个 *results.csv，返回 dict-of-lists。"""
    csvs = sorted(run_dir.glob("*results*.csv"))
    if not csvs:
        return None
    data = {"snr": [], "SSQ": [], "A": [], "B": [], "C": [], "D": []}
    with open(csvs[0], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data["snr"].append(float(row["snr_db"]))
            data["SSQ"].append(float(row["SSQ"]))
            data["A"].append(float(row["A_src_txt"]))
            data["B"].append(float(row["B_rec_txt"]))
            data["C"].append(float(row["C_orig_clf"]))
            data["D"].append(float(row["D_recon_clf"]))
    return data


def plot_ssq_comparison(runs: dict[str, dict], out: Path):
    """绘制多条 SSQ 曲线到一张图。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d in runs.items():
        st = STYLE.get(name, {"color": "gray", "marker": "x", "label": name})
        ax.plot(d["snr"], d["SSQ"], f'{st["marker"]}-', color=st["color"],
                linewidth=2.2, markersize=7, label=st["label"])
        for x, y in zip(d["snr"], d["SSQ"]):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel("SNR (dB)", fontsize=13)
    ax.set_ylabel("SSQ", fontsize=13)
    ax.set_title("Fig.7  AWGN Channel — SSQ vs SNR  (CatsVsDogs)", fontsize=14)
    ax.set_ylim(0.3, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"[OK] SSQ comparison saved: {out}")


def plot_abcd(name: str, d: dict, out: Path):
    """单个 run 的 ABCD 指标图。"""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(d["snr"], d["A"], "s--", color="#9333ea", linewidth=1.5, markersize=6, label="A (source text acc)")
    ax.plot(d["snr"], d["B"], "^-",  color="#ea580c", linewidth=2,   markersize=6, label="B (recovered text acc)")
    ax.plot(d["snr"], d["C"], "D--", color="#16a34a", linewidth=1.5, markersize=6, label="C (orig image clf)")
    ax.plot(d["snr"], d["D"], "o-",  color="#2563eb", linewidth=2,   markersize=6, label="D (recon image clf)")
    ax.set_xlabel("SNR (dB)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(f"Fig.7  ABCD vs SNR  ({STYLE.get(name, {}).get('label', name)})", fontsize=13)
    ax.set_ylim(50, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"[OK] ABCD plot saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None, help="指定要绘制的 run 名称")
    args = parser.parse_args()

    # 收集所有 run
    runs: dict[str, dict] = {}
    if args.run:
        d = _read_run(RESULTS_DIR / args.run)
        if d:
            runs[args.run] = d
    else:
        for sub in sorted(RESULTS_DIR.iterdir()):
            if sub.is_dir():
                d = _read_run(sub)
                if d:
                    runs[sub.name] = d

    if not runs:
        print("[WARN] No results CSV found in", RESULTS_DIR)
        return

    # SSQ 对比图（所有 run 在一张图上）
    plot_ssq_comparison(runs, RESULTS_DIR / "fig7_ssq_comparison.png")

    # 各个 run 的 ABCD 子图
    for name, d in runs.items():
        plot_abcd(name, d, RESULTS_DIR / name / f"fig7_abcd_{name}.png")

    plt.close("all")


if __name__ == "__main__":
    main()
