import argparse
import csv
import pathlib
from fractions import Fraction
from typing import Dict, List

import torch
import matplotlib.pyplot as plt

# Ensure package root is on sys.path
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import common  # 导入共享模块

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results" / "exp2"
LOG_DIR = RESULTS_DIR / "logs"


def save_metrics(rows: List[Dict[str, float]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "metrics.csv"
    fieldnames = ["train_snr", "test_snr", "mse", "psnr"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(rows: List[Dict[str, float]], train_snrs: List[float]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for t in train_snrs:
        xs = [r["test_snr"] for r in rows if r["train_snr"] == t]
        ys = [r["psnr"] for r in rows if r["train_snr"] == t]
        plt.plot(xs, ys, marker="o", label=f"train {t} dB")

    plt.xlabel("SNR_test (dB)")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs SNR_test (SNR mismatch)")
    plt.grid(True)
    plt.legend()
    plt.savefig(RESULTS_DIR / "psnr_vs_snr_test.png", dpi=150)
    plt.close()


def plot_loss_curve(loss_values: List[float], train_snr: float) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not loss_values:
        return
    plt.figure()
    xs = list(range(1, len(loss_values) + 1))
    plt.plot(xs, loss_values, linewidth=1.2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve (Train SNR={train_snr} dB)")
    plt.grid(True)
    filename = f"loss_curve_train_snr_{train_snr:g}dB.png"
    plt.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2: SNR mismatch robustness")
    
    # 添加通用参数
    common.add_common_args(parser)
    
    # 添加实验2特有参数
    parser.add_argument("--train-snrs", type=str, default="1,4,7,13,19", 
                        help="训练SNR列表（逗号分隔，论文Fig.4用[1,4,7,13,19]）")
    parser.add_argument("--test-snrs", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20", 
                        help="测试SNR范围（评估鲁棒性）")
    
    args = parser.parse_args()

    # 解析SNR列表
    train_snrs = [float(s.strip()) for s in args.train_snrs.split(",") if s.strip()]
    test_snrs = [float(s.strip()) for s in args.test_snrs.split(",") if s.strip()]

    # 计算编码器通道数
    kn = Fraction(args.kn)
    c = common.kn_to_c(kn)
    if c % 2 != 0:
        raise ValueError("Computed c must be even because channel pairs real/imag symbols.")

    kn_ratio = common.compute_kn_ratio(c)
    
    # 打印配置
    common.print_config(args, c, kn_ratio, "实验2: SNR Mismatch Robustness")
    print(f"训练SNR: {train_snrs}")
    print(f"测试SNR: {test_snrs}")

    device = torch.device(args.device)
    
    # 加载数据
    train_loader, test_loader = common.get_dataloaders(
        args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    # 训练和评估
    rows: List[Dict[str, float]] = []
    for train_snr in train_snrs:
        print(f"\n{'='*60}")
        print(f"  训练SNR={train_snr}dB的模型")
        print(f"{'='*60}")
        
        # 训练
        enc, dec, loss_history = common.train_one_snr(
            snr_db=train_snr,
            train_loader=train_loader,
            total_steps=args.steps,
            lr=args.lr,
            device=device,
            c=c,
            lr_schedule=args.lr_schedule,
            channel_type="awgn",
            desc_prefix=f"Train SNR={train_snr}",
            return_loss_history=True,
            log_dir=LOG_DIR,
        )

        plot_loss_curve(loss_history, train_snr)

        print(f"\n【测试SNR不匹配鲁棒性】训练SNR={train_snr}dB")
        for test_snr in test_snrs:
            # 评估
            mse, psnr = common.evaluate_model(
                enc=enc,
                dec=dec,
                snr_db=test_snr,
                test_loader=test_loader,
                device=device,
                repeats=args.repeats,
                channel_type="awgn"
            )
            
            rows.append({
                "train_snr": float(train_snr),
                "test_snr": float(test_snr),
                "mse": float(mse),
                "psnr": float(psnr),
            })
            print(f"  Train={train_snr}dB | Test={test_snr}dB | MSE={common.format_num(mse)} | PSNR={psnr:.3f}dB")

    # 保存结果
    save_metrics(rows)
    plot_curves(rows, train_snrs)

    print(f"\n{'='*60}")
    print(f"  实验2完成！结果保存到:")
    print(f"  {RESULTS_DIR}")
    print(f"  异常日志: {LOG_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
