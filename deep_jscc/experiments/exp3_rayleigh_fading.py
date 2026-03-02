"""实验3: Slow Rayleigh Fading (无CSI)

论文问题：在真实无线信道（衰落）下，Deep JSCC 是否更稳健？

信道模型：y = h·z + n
  - h ~ CN(0,1)，对整张图固定（block fading）
  - 训练：每个batch随机采样一个h
  - 测试：每张图随机一个h，多次传输取平均PSNR
  
关键假设：
  - 不给decoder显式CSI
  - 不传pilot
  - 网络自己"学会抗fading"
  
对比对象：JPEG/JPEG2000 + Channel Code（上界）
论文结论：数字方案→outage严重，Deep JSCC→表现稳定
"""

import argparse
import csv
import pathlib
from typing import Dict, List
from fractions import Fraction

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import common  # 导入共享模块

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results" / "exp3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ========== 实验3特有的函数 ==========

def save_metrics(rows: List[Dict[str, float]]) -> None:
    """保存指标到CSV"""
    csv_path = RESULTS_DIR / "metrics.csv"
    fieldnames = ["snr_db", "mse", "psnr"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Metrics saved to {csv_path}")


def plot_curves(metrics: List[Dict[str, float]]) -> None:
    """绘制PSNR vs SNR曲线"""
    snrs = [m["snr_db"] for m in metrics]
    psnrs = [m["psnr"] for m in metrics]
    
    plt.figure(figsize=(8, 6))
    plt.plot(snrs, psnrs, marker='o', label="Deep JSCC (Rayleigh)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("PSNR (dB)")
    plt.title("Experiment 3: Rayleigh Fading (No CSI)")
    plt.grid(True)
    plt.legend()
    plt.savefig(RESULTS_DIR / "psnr_vs_snr.png", dpi=150)
    plt.close()
    print(f"Plot saved to {RESULTS_DIR / 'psnr_vs_snr.png'}")


def save_visuals(model, snr: float, device: torch.device) -> None:
    """保存可视化样本（Rayleigh fading）"""
    enc, dec = model
    enc.eval()
    dec.eval()
    
    from model.channel import Channel
    channel = Channel(channel_type="rayleigh", snr_db=snr, fading="block")
    
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.CIFAR10(root=str(common.DATA_DIR), train=False, download=True, transform=transform)
    
    # 选取5张图
    imgs = []
    for i in range(5):
        img, _ = test_set[i]
        imgs.append(img)
    imgs = torch.stack(imgs).to(device)
    
    with torch.no_grad():
        z = enc(imgs)
        y = channel(z, snr_db=snr)
        recon = dec(y)
    
    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(5):
        # 原图
        axes[0, i].imshow(imgs[i].cpu().permute(1, 2, 0).numpy())
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=10)
        
        # 重建
        axes[1, i].imshow(recon[i].cpu().permute(1, 2, 0).numpy())
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title(f"Recon (SNR={snr}dB)", fontsize=10)
    
    plt.tight_layout()
    vis_path = RESULTS_DIR / "visual" / f"samples_snr{snr}.png"
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(vis_path, dpi=150)
    plt.close()
    print(f"Visual saved to {vis_path}")


# ========== 主函数 ==========

def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3: Rayleigh Fading (No CSI)")
    
    # 添加通用参数
    common.add_common_args(parser)
    
    # 添加实验3特有参数
    parser.add_argument("--snr", type=int, default=None, 
                        help="运行单个SNR实验（如10），默认运行[0,5,10,15,20]")
    
    args = parser.parse_args()

    # 计算编码器通道数
    kn = Fraction(args.kn)
    c = common.kn_to_c(kn)
    if c % 2 != 0:
        raise ValueError("Computed c must be even (channel pairs real/imag).")
    
    kn_ratio = common.compute_kn_ratio(c)
    
    # 打印配置
    common.print_config(args, c, kn_ratio, "实验3: Rayleigh Fading (No CSI)")
    
    device = torch.device(args.device)
    
    # 加载数据
    train_loader, test_loader = common.get_dataloaders(
        args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    
    # 确定要训练的SNR列表
    if args.snr is not None:
        snrs = [args.snr]
        print(f"【单SNR模式】只训练SNR={args.snr}dB")
    else:
        snrs = [0, 5, 10, 15, 20]
        print(f"【多SNR模式】训练SNR={snrs}")
    
    metrics: List[Dict[str, float]] = []
    
    for snr in snrs:
        print(f"\n{'='*60}")
        print(f"  训练SNR={snr}dB的模型 (Rayleigh Fading)")
        print(f"{'='*60}")
        
        # 训练（使用Rayleigh信道）
        enc, dec = common.train_one_snr(
            snr_db=snr,
            train_loader=train_loader,
            total_steps=args.steps,
            lr=args.lr,
            device=device,
            c=c,
            lr_schedule=args.lr_schedule,
            channel_type="rayleigh",
            desc_prefix=f"SNR={snr}dB (Rayleigh)"
        )
        
        # 评估（使用Rayleigh信道）
        print(f"\n【测试】SNR={snr}dB, {args.repeats}次重复传输")
        mse, psnr = common.evaluate_model(
            enc=enc,
            dec=dec,
            snr_db=snr,
            test_loader=test_loader,
            device=device,
            repeats=args.repeats,
            channel_type="rayleigh"
        )
        
        metrics.append({"snr_db": float(snr), "mse": float(mse), "psnr": float(psnr)})
        print(f"  SNR={snr}dB | MSE={common.format_num(mse)} | PSNR={psnr:.3f}dB")
        
        # 保存该SNR的可视化
        save_visuals((enc, dec), snr, device)
    
    # 保存结果
    save_metrics(metrics)
    plot_curves(metrics)
    
    print(f"\n{'='*60}")
    print(f"  实验3完成！结果保存到:")
    print(f"  {RESULTS_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
