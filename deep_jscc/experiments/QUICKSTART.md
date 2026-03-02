# 快速入门指南 - 重构后的代码使用

## 🚀 5分钟快速上手

### 1. 理解新的代码结构

```
experiments/
├── common.py                    # ⭐ 所有共享代码 (368行)
├── exp1_matched_snr.py          # 实验1 (164行, 减少63%)
├── exp2_snr_mismatch.py         # 实验2 (118行, 减少68%)
├── exp3_rayleigh_fading.py      # 实验3 (187行, 减少59%)
├── README_COMMON.md             # 详细文档
├── REFACTORING_SUMMARY.md       # 重构总结
└── BEFORE_AFTER_COMPARISON.md   # 对比示例
```

### 2. 运行实验 (使用方式不变)

```bash
# 实验1: Matched-SNR Training
python exp1_matched_snr.py --steps 100000 --snr 10

# 实验2: SNR Mismatch
python exp2_snr_mismatch.py --steps 100000

# 实验3: Rayleigh Fading
python exp3_rayleigh_fading.py --steps 100000
```

✅ **命令行参数完全兼容，无需修改现有脚本**

### 3. 修改通用参数 (现在超级简单)

**场景**: 想把所有实验的默认batch_size从128改为256

**之前** ❌: 需要在3个文件中分别修改
**现在** ✅: 只需修改 `common.py` 中的1处

```python
# common.py
def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=256, ...)  # 改这里
    # ✅ 所有实验自动使用新值
```

---

## 📖 常见修改任务

### 任务1: 修改默认学习率

**文件**: `common.py`
**位置**: `add_common_args()` 函数
**代码**:
```python
parser.add_argument("--lr", type=float, default=1e-4, ...)  # 改 1e-3 → 1e-4
```

### 任务2: 修改训练循环

**文件**: `common.py`
**位置**: `train_one_snr()` 函数
**示例**: 添加梯度累积
```python
def train_one_snr(..., accumulation_steps=1):
    for step in pbar:
        loss = loss / accumulation_steps
        loss.backward()
        if step % accumulation_steps == 0:
            optim.step()
            optim.zero_grad()
```

### 任务3: 修改评估逻辑

**文件**: `common.py`
**位置**: `evaluate_model()` 函数
**示例**: 添加更多指标
```python
def evaluate_model(...):
    # 现有代码...
    ssim = compute_ssim(x_hat, x)  # 添加SSIM计算
    return mse, psnr, ssim
```

---

## 🎯 关键优势

### ✅ "改一个就是改全部"

| 修改内容 | 重构前 | 重构后 |
|---------|-------|-------|
| 修改batch_size | 3个文件 | **1个文件** |
| 添加新参数 | 3个文件 × 3处 = 9处 | **1个文件 × 2处 = 2处** |
| 修复训练Bug | 3个文件 | **1个文件** |
| 添加GPU优化 | 3个文件 | **1个文件** |

**维护效率提升**: ~70%

### ✅ 代码质量保证

- 所有实验强制使用相同的训练逻辑
- 避免了代码不一致问题
- Bug修复一次即可

### ✅ 新功能添加容易

想要添加新的学习率策略？只需在 `common.py` 中添加一次。

---

## 📚 完整文档

- **基础使用**: [`README_COMMON.md`](README_COMMON.md)
- **重构总结**: [`REFACTORING_SUMMARY.md`](REFACTORING_SUMMARY.md)
- **对比示例**: [`BEFORE_AFTER_COMPARISON.md`](BEFORE_AFTER_COMPARISON.md)

---

## ⚡ 快速测试

验证重构是否成功：

```bash
# 快速测试所有实验 (每个只运行100步)
python exp1_matched_snr.py --steps 100 --snr 10
python exp2_snr_mismatch.py --steps 100 --train-snrs "10" --test-snrs "10"
python exp3_rayleigh_fading.py --steps 100 --snr 10
```

如果都能正常运行，说明重构成功！✅

---

## 🤔 常见问题

### Q: 我的旧脚本还能用吗？
A: 完全可以！命令行参数没有变化，旧脚本无需修改。

### Q: 我想修改某个实验特有的逻辑怎么办？
A: 直接修改对应的实验文件 (exp1/2/3_*.py)。共享逻辑在 `common.py`，特定逻辑在各自的实验文件中。

### Q: 如何添加新的实验？
A: 创建新的 `exp4_*.py` 文件，导入 `common` 模块，使用共享函数即可。

### Q: 我想用不同的训练逻辑怎么办？
A: `common.train_one_snr()` 支持 `channel_type` 参数，可以是 "awgn" 或 "rayleigh"。如需更多定制，可以在实验文件中重写。

---

## 💡 最佳实践

### ✅ DO (推荐)

- ✅ 修改共享逻辑时，修改 `common.py`
- ✅ 修改实验特定逻辑时，修改对应的实验文件
- ✅ 添加新参数时，考虑是否应该是共享参数
- ✅ 定期查看 `README_COMMON.md` 了解可用的共享函数

### ❌ DON'T (避免)

- ❌ 不要在实验文件中复制 `common.py` 的代码
- ❌ 不要在实验文件中重新实现已有的工具函数
- ❌ 不要在 `common.py` 中添加实验特定的逻辑
- ❌ 不要绕过 `common.add_common_args()` 自己定义通用参数

---

## 🎓 学习路径

1. **第1天**: 阅读本文档，了解基本结构
2. **第2天**: 运行所有实验，确认功能正常
3. **第3天**: 阅读 `BEFORE_AFTER_COMPARISON.md`，理解重构价值
4. **第4天**: 尝试修改 `common.py` 中的某个参数
5. **第5天**: 阅读 `common.py` 完整代码，理解所有共享函数

---

**快速入门完成！现在你可以高效地维护和扩展代码了。** 🎉
