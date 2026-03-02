import argparse
import pathlib
import itertools
import math
import copy
import datetime
from fractions import Fraction
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model.encoder import Encoder
from model.decoder import Decoder
from model.channel import Channel


# ========== 全局路径配置 ==========
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR.parent / "data"
LOG_DIR = BASE_DIR.parent / "logs"


# ========== 数据加载 ==========
def get_dataloaders(
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """获取CIFAR-10数据加载器（所有实验通用）
    
    Args:
        batch_size: 批次大小
        num_workers: 数据加载线程数（0=单线程，4=推荐GPU）
        pin_memory: 启用固定内存加速CPU→GPU传输
        persistent_workers: 保持worker进程活跃避免重建
    """
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_ds = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=transform)
    
    # GPU优化：多线程加载 + pin_memory加速传输 + persistent_workers避免重建
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=max(1, num_workers // 2),  # 测试用一半线程
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


# ========== 模型构建 ==========
def build_model(c: int) -> Tuple[Encoder, Decoder]:
    """构建编解码器（所有实验通用）
    
    Args:
        c: 编码器输出通道数（必须是偶数，用于I/Q配对）
    """
    enc = Encoder(in_channels=3, c=c)
    dec = Decoder(in_channels=c, out_channels=3)
    return enc, dec


# ========== 工具函数 ==========
def to_pixel_range(x: torch.Tensor) -> torch.Tensor:
    """将[0,1]转换到[0,255]用于PSNR计算"""
    return x * 255.0


def format_num(value: float) -> str:
    """格式化数字显示（避免不必要的科学计数法）"""
    abs_v = abs(value)
    if abs_v != 0.0 and (abs_v >= 1e7 or abs_v < 1e-6):
        return f"{value:.2e}"
    return f"{value:.6f}"


def compute_kn_ratio(c: int, h: int = 32, w: int = 32) -> float:
    """计算实际的k/n带宽压缩比
    
    Args:
        c: 编码器输出通道数
        h, w: 输入图像尺寸（CIFAR-10为32×32）
    
    Returns:
        k/n: 传输符号数/原始像素数
    """
    # Encoder下采样4倍（两次stride=2卷积）→ 8×8
    h_out, w_out = h // 4, w // 4
    # Channel使用复数（I/Q配对），每个复数占用c/2个通道
    k = (c // 2) * h_out * w_out
    n = 3 * h * w  # RGB三通道
    return k / n


def kn_to_c(kn: Fraction, h: int = 32, w: int = 32) -> int:
    """从k/n比率计算编码器通道数c
    
    Args:
        kn: 目标带宽压缩比（如Fraction("1/6")）
        h, w: 输入图像尺寸
    
    Returns:
        c: 编码器输出通道数（保证为偶数）
    """
    h_out, w_out = h // 4, w // 4
    n = 3 * h * w
    c_float = (2 * kn * n) / (h_out * w_out)
    c = int(round(float(c_float)))
    if c % 2 != 0:
        c += 1  # 确保c为偶数（I/Q配对需要）
    return c


# ========== 命令行参数解析器 ==========
def add_common_args(parser: argparse.ArgumentParser) -> None:
    """添加所有实验通用的命令行参数
    
    在各实验的main()中调用：
        parser = argparse.ArgumentParser(description="...")
        add_common_args(parser)
        # 再添加实验特定的参数...
        args = parser.parse_args()
    """
    # ========== 基础训练参数 ==========
    parser.add_argument("--steps", type=int, default=100000, help="训练总步数（每步处理一个batch）")
    # --steps: 训练总步数
    # default=100000: 默认10万步（论文用50万步）
    
    parser.add_argument("--batch-size", type=int, default=128, help="批次大小（越大GPU利用率越高）")
    # --batch-size: 每个batch包含的图片数量
    # default=128: 默认128（GPU优化值）
    
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率（控制参数更新步长）")
    # --lr: 学习率（learning rate）
    # default=1e-3: 默认0.001（Adam优化器常用值）
    
    parser.add_argument("--kn", type=str, default="1/6", help="带宽压缩比k/n（如1/6或1/12）")
    # --kn: 带宽压缩比
    # default="1/6": 默认1/6（传输数据量是原始数据的1/6）
    
    parser.add_argument("--repeats", type=int, default=10, help="测试时每张图重复传输次数（求平均）")
    # --repeats: 测试时重复次数
    # default=10: 默认10次传输求平均（因为信道有随机噪声）
    
    parser.add_argument("--device", type=str, default="cuda", help="计算设备（cuda或cpu）")
    # --device: 使用GPU还是CPU
    # default="cuda": 默认GPU加速
    
    parser.add_argument("--lr-schedule", type=str, default="warmup_cosine", choices=["test", "paper", "warmup_cosine", "adma"],
                        help="学习率衰减策略：test=自适应(60%%+plateau) | paper=固定(500k步) | warmup_cosine=热身+余弦衰减 | adma=仅使用Adam自适应(不额外调度)")
    # --lr-schedule: 学习率调度策略
    # "test": 60%步数时lr×0.1，loss停滞时lr×0.5（快速实验）
    # "paper": 50万步时lr×0.1（论文设置）
    
    # ========== GPU性能优化参数 ==========
    parser.add_argument("--num-workers", type=int, default=8, help="数据加载线程数（0=单线程，4-8=GPU推荐）")
    # --num-workers: 并行加载数据的线程数
    # default=4: 默认4线程（GPU训练时CPU同时准备下一批数据）
    
    parser.add_argument("--pin-memory", type=lambda x: str(x).lower() == 'true', default='true',
                        help="启用固定内存加速传输（true/false）")
    # --pin-memory: 固定内存（pinned memory）
    # default='true': 默认启用，加速CPU→GPU数据传输
    
    parser.add_argument("--persistent-workers", type=lambda x: str(x).lower() == 'true', default='true',
                        help="保持worker进程活跃避免重建（true/false）")
    # --persistent-workers: 持久化worker进程
    # default='true': 默认启用，训练更快但占用更多内存


# ========== 通用训练函数 ==========
def train_one_snr(
    snr_db: float,
    train_loader: DataLoader,
    total_steps: int,
    lr: float,
    device: torch.device,
    c: int,
    lr_schedule: str = "paper",
    channel_type: str = "awgn",
    desc_prefix: str = "SNR",
    return_loss_history: bool = False,
    log_dir: Optional[pathlib.Path] = None,
) -> Union[Tuple[Encoder, Decoder], Tuple[Encoder, Decoder, List[float]]]:
    """训练单个SNR模型（通用版本，支持AWGN和Rayleigh）
    
    Args:
        snr_db: 信噪比（dB）
        train_loader: 训练数据加载器
        total_steps: 训练总步数
        lr: 初始学习率
        device: 计算设备（cuda/cpu）
        c: 编码器通道数
        lr_schedule: 学习率调度策略（"test"/"paper"/"warmup_cosine"/"adma"）
        channel_type: 信道类型（"awgn"或"rayleigh"）
        desc_prefix: 进度条描述前缀
        return_loss_history: 是否返回loss历史
        log_dir: 异常日志保存目录（默认全局logs）
    
    Returns:
        (encoder, decoder): 训练好的模型
    """
    enc, dec = build_model(c)
    enc.to(device)
    dec.to(device)

    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr, eps=1e-4)

    enc.train()
    dec.train()

    ch = Channel(channel_type=channel_type, snr_db=float(snr_db)).to(device)
    # 优化进度条显示：自适应终端宽度，自定义格式
    snr_tag = f"{snr_db:g}"
    desc = desc_prefix if snr_tag in desc_prefix else f"{desc_prefix}={snr_db}dB"
    pbar = tqdm(total=total_steps, 
                desc=desc,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                mininterval=0.5)
    
    # 异常日志
    active_log_dir = log_dir if log_dir is not None else LOG_DIR
    active_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = active_log_dir / "anomaly_log.csv"
    if not log_path.exists():
        log_path.write_text("timestamp,event,snr_db,epoch,step,loss,loss_ema,lr,grad_norm\n", encoding="utf-8")

    def log_anomaly(event: str, step_idx: int, loss_val: float, lr_val: float, grad_val: float | None):
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        grad_str = "" if grad_val is None else f"{grad_val:.6f}"
        ema_str = "" if loss_ema is None else f"{loss_ema:.6f}"
        line = f"{ts},{event},{snr_db},{epoch+1},{step_idx},{loss_val:.6f},{ema_str},{lr_val:.6f},{grad_str}\n"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)

    # Loss plateau检测（使用EMA）
    loss_ema = None  # 指数移动平均
    best_loss_ema = float('inf')
    patience = 0
    patience_threshold = 50  # 连续50次检查无改进才降低lr（5000步）
    min_delta = 1e-4  # 最小改进阈值（提高以避免过度敏感）
    check_interval = 100  # 每100步检查一次
    lr_reduced_steps = []  # 记录lr降低的步数
    
    min_lr_abs = 1e-4  # 学习率最低值
    loss_history: List[float] = []
    total_steps_done = 0
    epoch_len = len(train_loader)
    total_epochs = max(1, math.ceil(total_steps / max(1, epoch_len)))
    epoch = 0

    while epoch < total_epochs and total_steps_done < total_steps:

        for xb, _ in train_loader:
            if total_steps_done >= total_steps:
                break
            xb = xb.to(device)
            
            # 前向传播
            z = enc(xb)
            y_noisy = ch(z)
            y = dec(y_noisy)
            loss = F.mse_loss(y, xb)

            # 异常检测：NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                current_lr = opt.param_groups[0]["lr"]
                log_anomaly("nan_or_inf", total_steps_done + 1, float("nan"), current_lr, None)
                print(f"\n[Anomaly] epoch={epoch+1} step={total_steps_done+1} loss=NaN/Inf (skip step)")
                continue
            
            # 反向传播
            opt.zero_grad()
            loss.backward()

            # 梯度裁剪（仅裁剪，不回退）
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(dec.parameters()),
                max_norm=1.0,
            )

            opt.step()
            
            # 获取loss值
            loss_value = loss.item()
            if return_loss_history:
                loss_history.append(loss_value)
            
            
            # 更新loss EMA（平滑系数0.1）
            loss_ema_prev = loss_ema
            if loss_ema is None:
                loss_ema = loss_value
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss_value
            
            # 仅裁剪，不回退

            # 学习率调度策略
            current_lr = opt.param_groups[0]["lr"]
            
            if lr_schedule == "paper":
                # 论文设置：在500k步降为1/10
                if total_steps_done + 1 == 500000:
                    for g in opt.param_groups:
                        g["lr"] = max(lr / 10.0, min_lr_abs)
                    current_lr = max(lr / 10.0, min_lr_abs)
                    lr_reduced_steps.append(total_steps_done + 1)
                    
            elif lr_schedule == "test":
                # 测试模式：基于步数比例和loss plateau
                
                # 策略1：在60%步数时降低学习率（只触发一次）
                if total_steps_done + 1 == int(total_steps * 0.6) and int(total_steps * 0.6) not in lr_reduced_steps:
                    for g in opt.param_groups:
                        g["lr"] = max(current_lr * 0.1, min_lr_abs)
                    current_lr = max(current_lr * 0.1, min_lr_abs)
                    lr_reduced_steps.append(total_steps_done + 1)
                    print(f"\n[Step {total_steps_done + 1}] Scheduled LR reduction to {current_lr:.2e}")
                
                # 策略2：基于EMA的plateau检测（每100步检查，但只在70%训练后启用，避免过早触发）
                if (total_steps_done + 1) % check_interval == 0 and (total_steps_done + 1) > total_steps * 0.7:
                    # 检查EMA是否有足够改进
                    if loss_ema < best_loss_ema - min_delta:
                        # 有改进，重置patience
                        best_loss_ema = loss_ema
                        patience = 0
                    else:
                        # 无明显改进，增加patience
                        patience += 1
                    
                    # 如果patience超过阈值且lr还能降低
                    if patience >= patience_threshold and current_lr > min_lr_abs:
                        for g in opt.param_groups:
                            g["lr"] = max(current_lr * 0.5, min_lr_abs)
                        current_lr = max(current_lr * 0.5, min_lr_abs)
                        lr_reduced_steps.append(total_steps_done + 1)
                        patience = 0  # 重置patience
                        best_loss_ema = loss_ema  # 重置最佳loss
                        print(f"\n[Step {total_steps_done + 1}] Loss plateau detected (EMA={loss_ema:.6f}), lr reduced to {current_lr:.2e}")

            elif lr_schedule == "warmup_cosine":
                # Warmup + Cosine衰减
                warmup_steps = max(1, int(total_steps * 0.05))  # 5%热身
                min_lr = max(lr * 0.01, 1e-5)  # 最低学习率
                step_index = total_steps_done + 1
                if step_index <= warmup_steps:
                    current_lr = lr * step_index / warmup_steps
                else:
                    progress = (step_index - warmup_steps) / max(1, total_steps - warmup_steps)
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    current_lr = min_lr + (lr - min_lr) * cosine
                for g in opt.param_groups:
                    g["lr"] = current_lr

            elif lr_schedule == "adma":
                # 仅使用Adam自适应更新，不做额外调度
                current_lr = max(current_lr, min_lr_abs)
                for g in opt.param_groups:
                    g["lr"] = current_lr

            total_steps_done += 1
            pbar.update(1)
            eq_epoch = total_steps_done * train_loader.batch_size / 50000.0
            pbar.set_postfix_str(f"ep={eq_epoch:.1f} loss={format_num(loss_value)} lr={current_lr:.1e} gn={float(grad_norm):.2f}")

        epoch += 1

    if return_loss_history:
        return enc, dec, loss_history
    return enc, dec


# ========== 通用评估函数 ==========
def evaluate_model(
    enc: Encoder,
    dec: Decoder,
    snr_db: float,
    test_loader: DataLoader,
    device: torch.device,
    repeats: int,
    channel_type: str = "awgn",
) -> Tuple[float, float]:
    """评估模型性能（通用版本，支持AWGN和Rayleigh）
    
    Args:
        enc, dec: 编解码器模型
        snr_db: 测试信噪比
        test_loader: 测试数据加载器
        device: 计算设备
        repeats: 每张图重复传输次数
        channel_type: 信道类型（"awgn"或"rayleigh"）
    
    Returns:
        (mse, psnr): 平均MSE和PSNR
    """
    enc.eval()
    dec.eval()

    total_mse = 0.0
    total_psnr = 0.0
    count = 0
    ch = Channel(channel_type=channel_type, snr_db=float(snr_db)).to(device)
    
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            mse_acc = torch.zeros(xb.shape[0], device=xb.device)
            
            for _ in range(repeats):
                y = dec(ch(enc(xb)))
                xb_px = to_pixel_range(xb)
                y_px = to_pixel_range(y)
                mse = F.mse_loss(y_px, xb_px, reduction="none")
                mse = mse.flatten(1).mean(dim=1)
                mse_acc += mse

            mse_mean = mse_acc / float(repeats)
            psnr = 10.0 * torch.log10((255.0 ** 2) / (mse_mean + 1e-12))
            total_mse += mse_mean.sum().item()
            total_psnr += psnr.sum().item()
            count += mse_mean.numel()

    mse = total_mse / max(1, count)
    psnr = total_psnr / max(1, count)
    return mse, psnr


# ========== 配置打印 ==========
def print_config(args, c: int, kn_ratio: float, exp_name: str = ""):
    """打印实验配置信息
    
    Args:
        args: 命令行参数对象
        c: 编码器通道数
        kn_ratio: 实际k/n比率
        exp_name: 实验名称（可选）
    """
    if exp_name:
        print(f"\n{'='*60}")
        print(f"  {exp_name}")
        print(f"{'='*60}")
    
    print(f"\n【模型配置】")
    print(f"  带宽压缩比: k/n={args.kn} → c={c} (实际k/n={kn_ratio:.6f})")
    
    print(f"\n【训练配置】")
    print(f"  训练步数: {args.steps:,} steps")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  学习率调度: {args.lr_schedule}")
    if args.lr_schedule == "test":
        print(f"    - LR将在第 {int(args.steps * 0.6):,} 步降低（60%）")
        print(f"    - LR将在loss停滞时自动降低")
    elif args.lr_schedule == "paper":
        print(f"    - LR将在第 500,000 步降低（论文设置）")
    elif args.lr_schedule == "warmup_cosine":
        print(f"    - 5%步数热身到初始LR")
        print(f"    - 余弦衰减到 LR×0.01")
    elif args.lr_schedule == "adma":
        print(f"    - 仅使用Adam自适应更新，不额外调度")
    
    print(f"\n【GPU配置】")
    print(f"  设备: {args.device}")
    print(f"  数据加载线程: {args.num_workers}")
    print(f"  固定内存(pin_memory): {args.pin_memory}")
    print(f"  持久化workers: {args.persistent_workers}")
    
    print(f"\n【测试配置】")
    print(f"  每图重复次数: {args.repeats}")
    print()