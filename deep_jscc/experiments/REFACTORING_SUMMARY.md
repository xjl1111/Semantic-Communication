# 代码重构总结 - "改一个就是改全部"

## 📊 重构成果

### 代码量对比

| 文件 | 重构前 | 重构后 | 减少行数 | 减少比例 |
|------|--------|--------|----------|----------|
| `exp1_matched_snr.py` | ~449行 | **164行** | 285行 | **63%** ↓ |
| `exp2_snr_mismatch.py` | ~370行 | **118行** | 252行 | **68%** ↓ |
| `exp3_rayleigh_fading.py` | ~455行 | **187行** | 268行 | **59%** ↓ |
| **共享模块** `common.py` | - | **368行** | - | - |
| **总计** | ~1274行 | **837行** | **437行** | **34%** ↓ |

### 文件大小对比

- `exp1_matched_snr.py`: **6 KB** (原约15 KB)
- `exp2_snr_mismatch.py`: **4.45 KB** (原约12 KB)
- `exp3_rayleigh_fading.py`: **6.3 KB** (原约16 KB)
- `common.py`: **14.42 KB** (新增)

**总文件大小**: 从 ~43 KB → **31.17 KB** (减少 27%)

---

## 🎯 实现的目标

### ✅ "改一个就是改全部"

现在所有共享代码都在 `common.py` 中，修改一次自动影响所有三个实验：

1. **数据加载优化** → 在 `common.get_dataloaders()` 中修改
2. **训练循环改进** → 在 `common.train_one_snr()` 中修改
3. **学习率策略调整** → 在 `common.train_one_snr()` 中修改
4. **评估逻辑更新** → 在 `common.evaluate_model()` 中修改
5. **工具函数增强** → 在 `common.py` 中修改

### ✅ 代码组织清晰

**共享模块 (`common.py`)**:
- ✅ 数据加载: `get_dataloaders()`
- ✅ 模型构建: `build_model()`
- ✅ 训练逻辑: `train_one_snr()` (支持 AWGN 和 Rayleigh)
- ✅ 评估逻辑: `evaluate_model()` (支持 AWGN 和 Rayleigh)
- ✅ 参数解析: `add_common_args()`
- ✅ 配置打印: `print_config()`
- ✅ 工具函数: `format_num()`, `to_pixel_range()`, `compute_kn_ratio()`, `kn_to_c()`

**实验文件** (只保留特定逻辑):
- `exp1_matched_snr.py`: 仅保留可视化函数 (`save_metrics`, `plot_curves`, `save_visuals`)
- `exp2_snr_mismatch.py`: 仅保留可视化函数 (`save_metrics`, `plot_curves`)
- `exp3_rayleigh_fading.py`: 仅保留可视化函数 (`save_metrics`, `plot_curves`, `save_visuals`)

---

## 🔧 使用方式

### 1. 修改所有实验的通用参数

**场景**: 想把默认batch_size从128改为256

```python
# 只需修改 common.py 中的一处
def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=256, ...)  # 改这里
    # ✅ 所有三个实验自动使用新默认值
```

### 2. 修改训练循环

**场景**: 想添加梯度累积

```python
# 只需修改 common.py 中的 train_one_snr()
def train_one_snr(...):
    # 在这里添加梯度累积逻辑
    # ✅ 所有三个实验自动获得新功能
```

### 3. 添加新的GPU优化

**场景**: 想启用自动混合精度 (AMP)

```python
# 只需修改 common.py
def train_one_snr(...):
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    # 添加AMP逻辑
    # ✅ 所有三个实验自动获得加速
```

---

## 📝 详细文档

完整使用指南请参考: [`README_COMMON.md`](README_COMMON.md)

---

## ✨ 关键优势

### 1. 维护成本大幅降低
- **之前**: 修改一个功能需要在3个文件中重复修改
- **现在**: 修改一次 `common.py` 即可

### 2. 代码一致性保证
- **之前**: 三个实验可能使用不同版本的相同函数
- **现在**: 所有实验强制使用统一的共享代码

### 3. 新功能添加容易
- **之前**: 新增功能需要在3个文件中分别实现
- **现在**: 在 `common.py` 中添加一次，所有实验自动获得

### 4. Bug修复效率提升
- **之前**: 发现bug需要在3个文件中分别修复
- **现在**: 在 `common.py` 中修复一次即可

---

## 🎓 最佳实践

### ✅ 应该放入 common.py 的代码

- ✅ 所有实验都使用的数据加载逻辑
- ✅ 通用的训练循环
- ✅ 通用的评估函数
- ✅ 工具函数 (格式化、转换等)
- ✅ 共享的参数定义

### ❌ 不应该放入 common.py 的代码

- ❌ 特定实验的可视化逻辑 (exp1的多SNR对比图)
- ❌ 特定实验的结果保存格式 (exp2的CSV列名)
- ❌ 特定实验的参数 (exp2的 `--train-snrs`)

---

## 🚀 快速验证

运行以下命令验证重构成功：

```bash
# 运行实验1 (快速测试)
python exp1_matched_snr.py --steps 100 --snr 10

# 运行实验2 (快速测试)
python exp2_snr_mismatch.py --steps 100 --train-snrs "5,10" --test-snrs "5,10,15"

# 运行实验3 (快速测试)
python exp3_rayleigh_fading.py --steps 100 --snr 10
```

所有实验应该正常运行，并使用 `common.py` 中的共享代码。

---

**重构完成日期**: 2024
**重构目标**: ✅ 实现"改一个就是改全部"的代码结构
**维护者**: 通过修改 `common.py` 即可影响所有实验
