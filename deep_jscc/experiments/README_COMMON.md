# 实验代码使用指南

## 📁 文件结构

```
experiments/
├── common.py              ← 【核心】所有实验的共享代码（修改这里影响全部实验）
├── exp1_matched_snr.py    ← 实验1：匹配SNR训练
├── exp2_snr_mismatch.py   ← 实验2：SNR不匹配鲁棒性
└── exp3_rayleigh_fading.py ← 实验3：Rayleigh衰落信道
```

## 🎯 核心思想

**修改一处，影响全部！**

所有公共代码（数据加载、模型构建、训练逻辑、参数解析等）都在 `common.py` 中。

### common.py 包含的功能

1. **数据加载**: `get_dataloaders()` - 统一的CIFAR-10加载器（含GPU优化）
2. **模型构建**: `build_model()` - 统一的编解码器构建
3. **训练函数**: `train_one_snr()` - 通用训练逻辑（支持AWGN和Rayleigh）
4. **评估函数**: `evaluate_model()` - 通用评估逻辑
5. **工具函数**: `format_num()`, `to_pixel_range()`, `compute_kn_ratio()`, `kn_to_c()`
6. **参数解析**: `add_common_args()` - 统一的命令行参数
7. **配置打印**: `print_config()` - 友好的配置信息显示

## 📝 使用方法

### 方式1：直接使用当前的实验文件

```bash
# 实验1
python exp1_matched_snr.py --steps 10000 --snr 20

# 实验2
python exp2_snr_mismatch.py --steps 10000 --train-snrs "1,4,7,13,19"

# 实验3
python exp3_rayleigh_fading.py --steps 10000 --snr 10
```

### 方式2：查看common.py了解全局配置

打开 `common.py` 查看和修改：
- 默认batch_size（当前128）
- 默认num_workers（当前4）
- 学习率调度策略
- 梯度裁剪阈值
- EMA平滑系数
- Patience阈值

**修改common.py后，所有实验自动生效！**

## ⚙️ 统一的命令行参数

所有实验支持以下参数（定义在`common.py`的`add_common_args()`）：

### 基础训练参数
- `--steps`: 训练总步数（默认100000）
- `--batch-size`: 批次大小（默认128，GPU优化）
- `--lr`: 学习率（默认1e-3）
- `--kn`: 带宽压缩比（默认"1/6"）
- `--repeats`: 测试时重复次数（默认10）
- `--device`: 计算设备（默认"cuda"）
- `--lr-schedule`: 学习率调度（"test"或"paper"，默认"test"）

### GPU性能优化参数
- `--num-workers`: 数据加载线程数（默认4）
- `--pin-memory`: 固定内存（默认true）
- `--persistent-workers`: 持久化workers（默认true）

## 🔧 如何修改全局配置

### 示例1：修改默认batch_size为256

编辑 `common.py`，找到：
```python
parser.add_argument("--batch-size", type=int, default=128, ...)
```
改为：
```python
parser.add_argument("--batch-size", type=int, default=256, ...)
```

保存后，所有实验默认使用256！

### 示例2：修改学习率调度策略

编辑 `common.py` 的 `train_one_snr()` 函数，调整：
- `patience_threshold = 20` → 改为30（更保守）
- `min_delta = 1e-5` → 改为1e-4（更宽容）
- EMA平滑系数 `0.9` → 改为0.95（更平滑）

### 示例3：添加新的优化技巧

在 `common.py` 的 `train_one_snr()` 中添加：
```python
# 添加学习率warmup
if step <= 1000:
    current_lr = lr * (step / 1000.0)
    for g in opt.param_groups:
        g["lr"] = current_lr
```

所有实验自动获得warmup功能！

## 📊 各实验特有的内容

虽然大部分代码共享，但每个实验仍保留其特有逻辑：

- **实验1**: `save_visuals()` - 可视化不同SNR的重建图像
- **实验2**: 多SNR交叉测试逻辑 - 训练一个SNR，测试所有SNR
- **实验3**: Rayleigh信道专用评估 - block fading模式

## 💡 最佳实践

1. **全局修改**：优先修改`common.py`
2. **实验特定修改**：只在各实验文件中修改其特有部分
3. **参数调优**：优先通过命令行参数调整，避免改代码
4. **代码同步**：确保`common.py`的函数签名稳定

## 🚀 快速开始

```bash
# 1. 快速测试（所有实验，小规模）
python exp1_matched_snr.py --steps 1000 --snr 20
python exp2_snr_mismatch.py --steps 1000 --train-snrs "1,7,13,19"
python exp3_rayleigh_fading.py --steps 1000 --snr 10

# 2. 完整实验（GPU，大规模）
python exp1_matched_snr.py --steps 100000 --batch-size 256
python exp2_snr_mismatch.py --steps 100000 --batch-size 256
python exp3_rayleigh_fading.py --steps 100000 --batch-size 256

# 3. CPU调试模式
python exp1_matched_snr.py --device cpu --num-workers 0 --pin-memory false --steps 100
```

## ⚠️ 注意事项

- **不要**在三个实验文件中重复修改相同的代码
- **优先**修改`common.py`来影响全局
- **测试**修改后用`--steps 100`快速验证
- **备份**重要的配置修改

---

**记住**：修改 `common.py` = 修改所有实验！
