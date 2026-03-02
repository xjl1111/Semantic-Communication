# Deep JSCC 实验代码 - 论文完全对齐版本

本代码库严格按照论文 **"Deep Joint Source–Channel Coding for Wireless Image Transmission"** (Bourtsoulatze et al., IEEE TCCN 2019) 实现。

## 📐 模型结构（已验证与论文图1完全一致）

### Encoder (5层卷积 + 功率归一化)
```
Input (3, 32, 32)
  ↓
5x5x16/2 conv + PReLU  → (16, 16, 16)
5x5x32/2 conv + PReLU  → (32, 8, 8)
5x5x32/1 conv + PReLU  → (32, 8, 8)
5x5x32/1 conv + PReLU  → (32, 8, 8)
5x5xc/1 conv + PReLU   → (c, 8, 8)
  ↓
Power Normalization (E[|z|²] = 1)
  ↓
Output: channel symbols
```

**关键修正**：
- ✅ 所有激活函数从ReLU改为**PReLU**（论文原文）
- ✅ 移除了错误的InstanceNorm（论文只在输出做功率归一化）
- ✅ 最后一层conv后包含PReLU，然后进行功率归一化

### Decoder (5层转置卷积)
```
Input (c, 8, 8)
  ↓
5x5x32/1 trans conv + PReLU → (32, 8, 8)
5x5x32/1 trans conv + PReLU → (32, 8, 8)
5x5x16/2 trans conv + PReLU → (16, 16, 16)
5x5x16/1 trans conv + PReLU → (16, 16, 16)
5x5x3/2 trans conv + Sigmoid → (3, 32, 32)
  ↓
Output: reconstructed image [0,1]
```

**关键修正**：
- ✅ 改为完整的5层转置卷积结构（之前只有3层conv+2层deconv）
- ✅ 确保两次stride=2上采样对应Encoder的两次下采样

### Channel Model
- **AWGN**: `y = z + n`, `n ~ CN(0, σ²)`, `σ² = 10^(-SNR/10) / 2`
- **Rayleigh Fading**: `y = h·z + n`
  - `h ~ CN(0, 1)` (每个实部/虚部 ~ N(0, 0.5))
  - **Block fading**: 整个batch共享一个h（实验3）
  - **Fast fading**: 每个符号独立h

## 🧪 实验配置

### 实验0: 基础环境验证（无训练）
**目的**: 验证模型实现正确性
- ✅ 形状接口测试
- ✅ Encoder功率归一化 (平均功率=1.0±5%)
- ✅ AWGN噪声理论一致性 (相对误差<5%)
- ✅ 未训练MSE与SNR无关性

**运行**:
```bash
python deep_jscc/tests/test_experiment0_sanity.py
```

### 实验1: Matched-SNR训练（AWGN信道）
**论文对应**: Fig. 3
**目的**: 建立baseline，证明end-to-end优化可行

**配置**:
- **Train SNRs**: 0, 5, 10, 20 dB
- **Test SNRs**: 与训练SNR相同（matched only）
- **k/n**: 1/12 (c=8) 或 1/6 (c=16)
- **Steps**: 100k (可调至500k)
- **Batch**: 64
- **LR**: 1e-3 → 1e-4 (at 500k steps)
- **Repeats**: 10次/图像

**运行**:
```bash
# k/n=1/12, 快速测试
python deep_jscc/experiments/exp1_matched_snr.py --kn 1/12 --steps 10000

# k/n=1/6, 完整训练
python deep_jscc/experiments/exp1_matched_snr.py --kn 1/6 --steps 500000 --device cuda
```

**输出**:
- `results/exp1/metrics.csv`: SNR, MSE, PSNR
- `results/exp1/psnr_vs_snr.png`: PSNR曲线
- `results/exp1/visual/`: 可视化重建样本

### 实验2: SNR Mismatch / 鲁棒性（AWGN信道）
**论文对应**: Fig. 4(a) 和 4(b)
**目的**: 展示Deep JSCC的优雅降级（无cliff effect）

**配置**:
- **Train SNRs**: **1, 4, 7, 13, 19 dB** （论文图4原始值）
- **Test SNRs**: 0-20 dB (每1dB采样)
- **k/n**: 
  - 1/12 (c=8) for Fig.4(a)
  - 1/6 (c=16) for Fig.4(b)
- **Steps**: 100k/SNR
- **Batch**: 64
- **Repeats**: 10次/图像

**运行**:
```bash
# 复现Fig.4(a) - k/n=1/12
python deep_jscc/experiments/exp2_snr_mismatch.py --kn 1/12 --steps 100000

# 复现Fig.4(b) - k/n=1/6
python deep_jscc/experiments/exp2_snr_mismatch.py --kn 1/6 --steps 100000 --device cuda
```

**输出**:
- `results/exp2/metrics.csv`: train_snr, test_snr, mse, psnr
- `results/exp2/psnr_vs_snr_test.png`: 多条曲线（每个train_snr一条）

**预期结果**:
- 低SNR训练的模型：低SNR测试好，高SNR测试饱和
- 高SNR训练的模型：高SNR测试好，低SNR测试差
- **无cliff effect**：曲线平滑过渡（vs 数字方案的突变）

### 实验3: Rayleigh Fading（无CSI）
**论文对应**: Section V-B
**目的**: 验证Deep JSCC在真实衰落信道的稳健性

**信道模型**:
```
y = h·z + n
h ~ CN(0, 1), 对整张图固定（block fading）
```

**配置**:
- **SNRs**: 0, 5, 10, 15, 20 dB
- **k/n**: 1/12
- **Steps**: 100k/SNR
- **Repeats**: 10次/图像 (每次随机新的h)
- **关键**: 
  - 训练时每batch随机h
  - 测试时每图随机h，多次传输平均
  - **不给decoder CSI**，网络自己学会抗衰落

**运行**:
```bash
python deep_jscc/experiments/exp3_rayleigh_fading.py --kn 1/12 --steps 100000
```

**输出**:
- `results/exp3/metrics.csv`: SNR, MSE, PSNR
- `results/exp3/psnr_vs_snr.png`: PSNR vs SNR曲线
- `results/exp3/visual/`: 每个SNR的重建样本

## 📊 关键公式

### 压缩比 k/n
对于CIFAR-10 (32×32 RGB):
```
k = (c × 8 × 8) / 2    # 复数符号数
n = 32 × 32 × 3         # 像素总数
k/n = c / 96
```
- k/n = 1/12 → c = 8
- k/n = 1/6 → c = 16

### PSNR计算
```python
MSE = mean((img_pixel - recon_pixel)²)  # 像素域[0,255]
PSNR = 10 × log10(255² / MSE)
```

**重要**: 
- 训练损失在[0,1]域计算MSE
- PSNR在[0,255]域计算（MAX=255）
- 测试时先per-image MSE，再平均PSNR

## ✅ 与论文完全对齐的检查清单

- [x] **数据预处理**: 仅ToTensor（[0,1]），无mean/std归一化
- [x] **Encoder激活**: PReLU（非ReLU）
- [x] **Encoder层数**: 5层conv + 功率归一化
- [x] **Decoder层数**: 5层trans conv（非3层）
- [x] **功率归一化**: 仅在Encoder输出，约束E[|z|²]=1
- [x] **信道模型**: AWGN和Rayleigh完全按论文实现
- [x] **PSNR公式**: MAX=255，per-image averaging
- [x] **训练参数**: batch=64, lr=1e-3→1e-4, steps可调
- [x] **实验1 SNRs**: 0,5,10,20 dB (matched only)
- [x] **实验2 SNRs**: train=[1,4,7,13,19], test=[0-20]
- [x] **实验3**: Block fading, 无CSI, 多次传输平均
- [x] **k/n计算**: c/96 for 32×32 images

## 🚀 快速开始

### 完整测试流程
```bash
# 1. 验证环境（应全部PASS）
python deep_jscc/tests/test_experiment0_sanity.py

# 2. 快速实验（小steps验证代码）
python deep_jscc/experiments/exp1_matched_snr.py --kn 1/12 --steps 1000 --snr 10
python deep_jscc/experiments/exp2_snr_mismatch.py --kn 1/12 --steps 1000
python deep_jscc/experiments/exp3_rayleigh_fading.py --kn 1/12 --steps 1000 --snr 10

# 3. 完整实验（需GPU，耗时数小时）
python deep_jscc/experiments/exp1_matched_snr.py --kn 1/12 --steps 500000 --device cuda
python deep_jscc/experiments/exp2_snr_mismatch.py --kn 1/12 --steps 100000 --device cuda
python deep_jscc/experiments/exp3_rayleigh_fading.py --kn 1/12 --steps 100000 --device cuda
```

## 📝 代码修正历史

### 2026-01-25 全面对齐论文
1. **模型结构修正**:
   - Encoder: ReLU → PReLU, 移除InstanceNorm
   - Decoder: 3层conv+2层deconv → 5层trans conv
   
2. **实验配置修正**:
   - 实验1: 移除cross-SNR评估（属于实验2）
   - 实验2: train_snrs改为论文Fig.4的[1,4,7,13,19]
   - 实验3: 新增Rayleigh fading实验

3. **训练参数调整**:
   - 默认steps从5k增至100k（更合理）
   - 确认batch=64, lr schedule, repeats=10

## 🎯 预期性能

基于论文Fig.3和Fig.4:

| k/n  | SNR (dB) | PSNR (dB) | 备注 |
|------|----------|-----------|------|
| 1/12 | 0        | ~22       | 低SNR matched |
| 1/12 | 10       | ~28       | 中SNR matched |
| 1/12 | 20       | ~31       | 高SNR matched |
| 1/6  | 0        | ~24       | 低SNR, 更高k/n |
| 1/6  | 10       | ~31       | 中SNR, 更高k/n |
| 1/6  | 20       | ~36       | 高SNR, 更高k/n |

**注意**: 实际性能受训练steps、初始化、数据集划分等影响，可能有±2dB浮动。

## 📖 参考文献

```bibtex
@article{bourtsoulatze2019deep,
  title={Deep Joint Source--Channel Coding for Wireless Image Transmission},
  author={Bourtsoulatze, Eirina and Kurka, David Burth and Gunduz, Deniz},
  journal={IEEE Transactions on Cognitive Communications and Networking},
  volume={5},
  number={3},
  pages={567--579},
  year={2019},
  publisher={IEEE}
}
```

## ⚠️ 重要提醒

1. **训练时间**: 500k steps约需数小时（GPU）或数天（CPU）
2. **显存需求**: batch=64, c=16约需4GB显存
3. **数据集**: 首次运行会自动下载CIFAR-10 (~170MB)
4. **结果复现**: 由于随机性，PSNR可能与论文有±1-2dB偏差
5. **调试技巧**: 先用--steps 1000快速验证，再full training

---

**代码状态**: ✅ 已完全对齐论文，通过实验0验证
**最后更新**: 2026-01-25
**维护者**: Deep JSCC研究组
