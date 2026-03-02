# 学习率调度策略说明

## 新增参数

```bash
--lr-schedule {test,paper}
```

## 两种模式

### 1. `--lr-schedule test`（测试模式，默认）

**适合快速实验和调试**，自动适应您指定的训练步数：

**策略组合**：
- **基于步数**：在总步数的60%时自动降低学习率至1/10
  - 例如：10000步训练 → 在第6000步降低lr
  - 例如：50000步训练 → 在第30000步降低lr

- **基于Loss Plateau**：
  - 每100步检查loss趋势
  - 如果最近500步loss改进<1%，且已过30%训练进度
  - 连续3次检测到plateau → 学习率减半
  - 防止陷入局部最优

**示例**：
```bash
# 10000步训练，自动在6000步降低lr
python deep_jscc/experiments/exp1_matched_snr.py --steps 10000 --lr-schedule test --snr 20

# 50000步训练，自动在30000步降低lr
python deep_jscc/experiments/exp1_matched_snr.py --steps 50000 --lr-schedule test --kn 1/12
```

### 2. `--lr-schedule paper`（论文模式）

**严格按照论文设置**：
- 在第500000步时学习率降为1/10
- 适合完整复现论文实验

**示例**：
```bash
# 完整论文实验（500k步）
python deep_jscc/experiments/exp1_matched_snr.py --steps 500000 --lr-schedule paper --device cuda
```

## 进度条显示

现在进度条会实时显示学习率：

```
SNR=20 dB: 100%|████| 10000/10000 [05:09<00:00, 32.27it/s, eq_epoch=12.80, loss=0.019314, lr=1.00e-04, step=1e+4]
                                                            ^^^^^^^^^^^^^^^^         ^^^^^^^^^^^
                                                            当前loss                  当前学习率
```

## 学习率调整时的提示

当学习率降低时，会在终端打印提示：

```
[Step 6000] Learning rate reduced to 1.00e-04

或

[Step 7800] Loss plateau detected, lr reduced to 5.00e-05
```

## 使用建议

### 快速测试（推荐test模式）
```bash
# 10k步快速测试
python deep_jscc/experiments/exp1_matched_snr.py --steps 10000 --lr-schedule test --snr 20
```

### 中等规模训练
```bash
# 100k步训练，自动在60k步降低lr
python deep_jscc/experiments/exp1_matched_snr.py --steps 100000 --lr-schedule test --kn 1/12
```

### 完整论文复现
```bash
# 500k步，使用论文lr调度
python deep_jscc/experiments/exp1_matched_snr.py --steps 500000 --lr-schedule paper --device cuda
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr-schedule` | `test` | `test`: 自适应调度, `paper`: 论文固定调度 |
| `--steps` | `10000` | 总训练步数 |
| `--lr` | `1e-3` | 初始学习率 |

## Test模式的详细逻辑

### 步数降低（60%规则）
```python
if step == int(total_steps * 0.6):
    lr = lr * 0.1  # 降为1/10
```

### Plateau检测（自适应）
```python
每100步检查:
    if 最近500步平均loss >= 之前500步平均loss * 0.99:
        patience_counter += 1
        if patience_counter >= 3:  # 连续3次plateau
            lr = lr * 0.5  # 减半
```

## 解决您的问题

您遇到的**第8000步loss突然跳跃**问题：

1. **梯度裁剪已添加** → 防止梯度爆炸
2. **Test模式会在6000步降低lr** → 更稳定的训练
3. **Plateau检测自动调整** → 如果loss不降会主动降低lr
4. **实时显示lr** → 随时监控学习率变化

**现在重新运行**：
```bash
python deep_jscc/experiments/exp1_matched_snr.py --steps 10000 --lr-schedule test --snr 20
```

您应该看到：
- 第6000步：`[Step 6000] Learning rate reduced to 1.00e-04`
- 进度条显示：`lr=1.00e-03` → `lr=1.00e-04`
- Loss平滑下降，不会突然跳跃

---

**修改时间**: 2026-01-25  
**解决问题**: Loss突然跳跃 + 学习率可视化 + 灵活调度策略
