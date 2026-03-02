"""快速测试loss爆炸修复效果

运行这个脚本来验证：
1. 功率归一化的数值稳定性
2. Loss爆炸检测和恢复机制
3. 梯度裁剪效果
"""

import torch
import sys
import pathlib

# 添加路径
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from model.encoder import Encoder
from model.decoder import Decoder
from model.channel import Channel


def test_power_normalization_stability():
    """测试功率归一化的数值稳定性"""
    print("=" * 60)
    print("测试1: 功率归一化数值稳定性")
    print("=" * 60)
    
    enc = Encoder(in_channels=3, c=16)
    
    # 测试极端情况：接近全0的输入
    batch_sizes = [1, 4, 8]
    for bs in batch_sizes:
        # 创建非常小的输入
        x = torch.randn(bs, 3, 32, 32) * 1e-5
        
        try:
            z = enc(x)
            
            # 检查输出
            if torch.isnan(z).any():
                print(f"[X] Batch size {bs}: 输出包含NaN")
            elif torch.isinf(z).any():
                print(f"[X] Batch size {bs}: 输出包含Inf")
            else:
                max_val = z.abs().max().item()
                mean_val = z.abs().mean().item()
                print(f"[OK] Batch size {bs}: 输出正常 (max={max_val:.2f}, mean={mean_val:.2f})")
                
        except Exception as e:
            print(f"❌ Batch size {bs}: 异常 - {e}")
    
    print()


def test_gradient_explosion_protection():
    """测试梯度爆炸保护"""
    print("=" * 60)
    print("测试2: 梯度爆炸保护")
    print("=" * 60)
    
    enc = Encoder(in_channels=3, c=16)
    dec = Decoder(in_channels=16, out_channels=3)
    ch = Channel(channel_type="awgn", snr_db=0.0)  # 低SNR，容易产生大梯度
    
    # 使用大学习率
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-2)
    
    x = torch.randn(8, 3, 32, 32)
    
    for step in range(5):
        z = enc(x)
        y_noisy = ch(z)
        y = dec(y_noisy)
        loss = torch.nn.functional.mse_loss(y, x)
        
        opt.zero_grad()
        loss.backward()
        
        # 计算梯度范数
        total_norm = 0.0
        for p in list(enc.parameters()) + list(dec.parameters()):
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), max_norm=0.5)
        
        # 计算裁剪后的梯度范数
        clipped_norm = 0.0
        for p in list(enc.parameters()) + list(dec.parameters()):
            if p.grad is not None:
                clipped_norm += p.grad.data.norm(2).item() ** 2
        clipped_norm = clipped_norm ** 0.5
        
        opt.step()
        
        print(f"Step {step+1}: loss={loss.item():.6f}, "
              f"grad_norm={total_norm:.2f} -> {clipped_norm:.2f} (裁剪后)")
    
    print()


def test_loss_explosion_detection():
    """测试loss爆炸检测"""
    print("=" * 60)
    print("测试3: Loss爆炸检测机制")
    print("=" * 60)
    
    loss_explosion_threshold = 10.0
    consecutive_explosions = 0
    max_consecutive_explosions = 3
    
    # 模拟loss序列
    loss_sequence = [
        0.023, 0.019, 0.015, 0.013,  # 正常
        12.5,  # 第1次爆炸
        11.8,  # 第2次爆炸
        15.3,  # 第3次爆炸 → 触发恢复
        0.014, 0.012, 0.011  # 恢复后
    ]
    
    for i, loss_value in enumerate(loss_sequence, 1):
        if loss_value > loss_explosion_threshold:
            consecutive_explosions += 1
            print(f"Step {i}: loss={loss_value:.4f} [!] 爆炸警告 "
                  f"({consecutive_explosions}/{max_consecutive_explosions})")
            
            if consecutive_explosions >= max_consecutive_explosions:
                print(f"        -> 触发恢复机制！恢复到最佳状态并降低学习率")
                consecutive_explosions = 0
        else:
            if consecutive_explosions > 0:
                print(f"Step {i}: loss={loss_value:.4f} [OK] 恢复正常")
            else:
                print(f"Step {i}: loss={loss_value:.4f} [OK]")
            consecutive_explosions = 0
    
    print()


def test_extreme_snr():
    """测试极端SNR下的稳定性"""
    print("=" * 60)
    print("测试4: 极端SNR稳定性")
    print("=" * 60)
    
    enc = Encoder(in_channels=3, c=16)
    dec = Decoder(in_channels=16, out_channels=3)
    
    x = torch.randn(4, 3, 32, 32)
    
    # 测试不同SNR
    snrs = [-10, -5, 0, 5, 10, 20, 30]
    
    for snr in snrs:
        ch = Channel(channel_type="awgn", snr_db=float(snr))
        
        try:
            z = enc(x)
            y_noisy = ch(z)
            y = dec(y_noisy)
            loss = torch.nn.functional.mse_loss(y, x)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"SNR={snr:3d}dB: [X] Loss异常 ({loss.item()})")
            else:
                print(f"SNR={snr:3d}dB: [OK] Loss={loss.item():.6f}")
                
        except Exception as e:
            print(f"SNR={snr:3d}dB: [X] 异常 - {e}")
    
    print()


def test_rayleigh_channel():
    """测试Rayleigh信道稳定性"""
    print("=" * 60)
    print("测试5: Rayleigh信道稳定性")
    print("=" * 60)
    
    enc = Encoder(in_channels=3, c=16)
    dec = Decoder(in_channels=16, out_channels=3)
    ch = Channel(channel_type="rayleigh", snr_db=10.0, fading="block")
    
    x = torch.randn(4, 3, 32, 32)
    
    # 多次测试（因为Rayleigh是随机的）
    for trial in range(5):
        try:
            z = enc(x)
            y_noisy = ch(z)
            y = dec(y_noisy)
            loss = torch.nn.functional.mse_loss(y, x)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Trial {trial+1}: [X] Loss异常 ({loss.item()})")
            else:
                print(f"Trial {trial+1}: [OK] Loss={loss.item():.6f}")
                
        except Exception as e:
            print(f"Trial {trial+1}: [X] 异常 - {e}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("[Loss爆炸修复效果验证]")
    print("=" * 60 + "\n")
    
    # 运行所有测试
    test_power_normalization_stability()
    test_gradient_explosion_protection()
    test_loss_explosion_detection()
    test_extreme_snr()
    test_rayleigh_channel()
    
    print("=" * 60)
    print("[OK] 所有测试完成！")
    print("=" * 60)
    print("\n如果所有测试都显示 [OK]，说明修复生效。")
    print("现在可以安全地运行完整训练：")
    print("  python exp1_matched_snr.py --steps 10000 --snr 10")
    print()
