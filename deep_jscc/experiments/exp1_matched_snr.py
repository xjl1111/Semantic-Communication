"""实验1: Matched-SNR训练（使用共享模块common.py）

每个SNR训练独立的模型，测试时使用相同的SNR。
目标：验证Deep JSCC在不同SNR下的基本性能。

【简化版本】大部分代码在common.py中，这里只保留实验特有的逻辑：
- save_metrics(): 保存CSV结果
- plot_curves(): 绘制PSNR/MSE曲线
- save_visuals(): 保存重建图像可视化
"""

import argparse
import csv
import pathlib
from fractions import Fraction
from typing import Dict, List, Tuple

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 导入共享模块（包含所有通用代码）
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import common
from model.channel import Channel


RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results" / "exp1"
VIS_DIR = RESULTS_DIR / "visual"


# ========== 实验1特有的函数（绘图和可视化） ==========

def save_metrics(metrics: List[Dict[str, float]]) -> None:
    """保存指标到CSV文件"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["snr_db", "mse", "psnr"])
        writer.writeheader()
        writer.writerows(metrics)


def plot_curves(metrics: List[Dict[str, float]]) -> None:
    """绘制PSNR和MSE曲线"""
    snrs = [m["snr_db"] for m in metrics]
    mses = [m["mse"] for m in metrics]
    psnrs = [m["psnr"] for m in metrics]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # PSNR曲线
    plt.figure()
    plt.plot(snrs, psnrs, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs SNR (Matched)")
    plt.grid(True)
    plt.savefig(RESULTS_DIR / "psnr_vs_snr.png", dpi=150)
    plt.close()

    # MSE曲线
    plt.figure()
    plt.plot(snrs, mses, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("MSE")
    plt.title("MSE vs SNR (Matched)")
    plt.grid(True)
    plt.savefig(RESULTS_DIR / "mse_vs_snr.png", dpi=150)
    plt.close()


def plot_loss_curve(loss_values: List[float], snr: float) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not loss_values:
        return
    plt.figure()
    xs = list(range(1, len(loss_values) + 1))
    plt.plot(xs, loss_values, linewidth=1.2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve (SNR={snr} dB)")
    plt.grid(True)
    plt.savefig(RESULTS_DIR / f"loss_curve_snr_{snr:g}dB.png", dpi=150)
    plt.close()


def save_visuals(
    models: Dict[float, Tuple],
    snrs: List[float],
    device: torch.device,
) -> None:
    """保存不同SNR下的重建图像可视化"""
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.CIFAR10(root=str(common.DATA_DIR), train=False, download=True, transform=transform)
    x0, _ = test_ds[0]
    x0 = x0.unsqueeze(0).to(device)

    # 保存原图
    orig = x0.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    plt.imsave(VIS_DIR / "orig.png", orig)

    # 保存每个SNR的重建图
    for snr in snrs:
        enc, dec = models[snr]
        enc.eval()
        dec.eval()
        with torch.no_grad():
            ch = Channel(channel_type="awgn", snr_db=float(snr)).to(device)
            x_hat = dec(ch(enc(x0)))
        img = x_hat.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
        plt.imsave(VIS_DIR / f"recon_snr_{int(snr)}.png", img)


# ========== 主函数 ==========

def main() -> None:
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Experiment 1: Matched-SNR training over AWGN")
    
    # 添加通用参数（来自common.py）
    common.add_common_args(parser)
    
    # 添加实验1特有的参数
    parser.add_argument("--snr", type=int, default=None, help="运行单个SNR实验（如10），None默认运行[0,5,10,20]")
    
    args = parser.parse_args()

    # 计算编码器通道数
    kn = Fraction(args.kn)
    c = common.kn_to_c(kn)
    if c % 2 != 0:
        raise ValueError("Computed c must be even because channel pairs real/imag symbols.")
    
    kn_ratio = common.compute_kn_ratio(c)
    
    # 打印配置信息
    common.print_config(args, c, kn_ratio, "实验1: Matched-SNR Training")
    
    # 确定要训练的SNR列表
    device = torch.device(args.device)
    snrs = [0, 5, 10, 20]
    if args.snr is not None:
        snrs = [int(args.snr)]
        print(f"【单SNR模式】只训练SNR={args.snr}dB")
    else:
        print(f"【多SNR模式】训练SNR={snrs}")

    # 加载数据（使用common.py的函数）
    train_loader, test_loader = common.get_dataloaders(
        args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    # 训练和评估
    metrics: List[Dict[str, float]] = []
    models: Dict[float, Tuple] = {}

    for snr in snrs:
        print(f"\n{'='*60}")
        print(f"  训练SNR={snr}dB的模型")
        print(f"{'='*60}")
        
        # 训练（使用common.py的通用训练函数）
        enc, dec, loss_history = common.train_one_snr(
            snr_db=snr,
            train_loader=train_loader,
            total_steps=args.steps,
            lr=args.lr,
            device=device,
            c=c,
            lr_schedule=args.lr_schedule,
            channel_type="awgn",
            desc_prefix="SNR",
            return_loss_history=True,
        )

        plot_loss_curve(loss_history, snr)
        
        # 评估（使用common.py的通用评估函数）
        mse, psnr = common.evaluate_model(
            enc=enc,
            dec=dec,
            snr_db=snr,
            test_loader=test_loader,
            device=device,
            repeats=args.repeats,
            channel_type="awgn"
        )
        
        metrics.append({"snr_db": float(snr), "mse": float(mse), "psnr": float(psnr)})
        models[float(snr)] = (enc, dec)
        
        print(f"\n【测试结果】SNR={snr}dB | MSE={common.format_num(mse)} | PSNR={psnr:.3f}dB")

    # 保存结果（实验1特有）
    save_metrics(metrics)
    plot_curves(metrics)
    save_visuals(models, snrs, device)

    print(f"\n{'='*60}")
    print(f"  实验1完成！结果保存到:")
    print(f"  {RESULTS_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
