# 重构前后对比示例

## 场景1：修改默认batch_size

### ❌ 重构前 (需要修改3个文件)

**exp1_matched_snr.py**:
```python
parser.add_argument("--batch-size", type=int, default=128, ...)
```

**exp2_snr_mismatch.py**:
```python
parser.add_argument("--batch-size", type=int, default=128, ...)
```

**exp3_rayleigh_fading.py**:
```python
parser.add_argument("--batch-size", type=int, default=128, ...)
```

**问题**: 
- 需要在3个文件中分别修改
- 容易遗漏某个文件
- 可能导致不一致

---

### ✅ 重构后 (只需修改1个文件)

**common.py**:
```python
def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=128, ...)  # 改这里
    # ✅ 所有3个实验自动使用新值
```

**优势**:
- ✅ 只修改1次
- ✅ 不会遗漏
- ✅ 保证一致性

---

## 场景2：添加梯度裁剪阈值参数

### ❌ 重构前

需要在3个文件的 `train_one_snr()` 函数中分别添加：
1. 参数定义
2. 梯度裁剪逻辑
3. 参数解析

**工作量**: 3个文件 × 3处修改 = **9处修改**

---

### ✅ 重构后

**common.py** (只需2处修改):
```python
def add_common_args(parser):
    parser.add_argument("--grad-clip", type=float, default=1.0, ...)  # 1. 添加参数

def train_one_snr(..., grad_clip=1.0):
    torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)  # 2. 使用参数
```

**工作量**: 1个文件 × 2处修改 = **2处修改**

**效率提升**: 77.8% (从9处减少到2处)

---

## 场景3：修复训练循环中的Bug

### ❌ 重构前

**发现bug**: 学习率衰减逻辑有问题

需要在以下3个文件中分别修复：
- `exp1_matched_snr.py` 的 `train_one_snr()` (约200行函数)
- `exp2_snr_mismatch.py` 的 `train_one_snr()` (约200行函数)
- `exp3_rayleigh_fading.py` 的 `train_one_snr()` (约200行函数)

**风险**:
- ❌ 可能在某个文件中修复不完整
- ❌ 可能引入新的不一致性
- ❌ 测试需要运行3次

---

### ✅ 重构后

**修复bug**: 只需在 `common.py` 的 `train_one_snr()` 中修复一次

**优势**:
- ✅ 修复1次即可
- ✅ 保证所有实验获得修复
- ✅ 测试1次即可验证

---

## 场景4：添加新的学习率策略

### ❌ 重构前

**需求**: 添加 "cosine" 学习率衰减策略

**实现步骤**:
1. 在 `exp1_matched_snr.py` 中:
   - 添加参数选项
   - 实现cosine逻辑
   - 测试
2. 在 `exp2_snr_mismatch.py` 中:
   - 复制粘贴代码
   - 测试
3. 在 `exp3_rayleigh_fading.py` 中:
   - 复制粘贴代码
   - 测试

**总时间**: 约3小时 (每个文件1小时)

---

### ✅ 重构后

**实现步骤**:
1. 在 `common.py` 中:
   - 添加 "cosine" 到 `--lr-schedule` 选项
   - 实现cosine逻辑
   - 测试一次

**总时间**: 约1小时

**效率提升**: 66.7%

---

## 场景5：启用自动混合精度 (AMP)

### ❌ 重构前

需要在3个文件中分别修改：
1. 导入 `torch.cuda.amp`
2. 创建 `GradScaler`
3. 修改前向传播代码
4. 修改反向传播代码
5. 添加 `--amp` 参数

**总修改**: 3个文件 × 5处 = **15处修改**

---

### ✅ 重构后

**common.py** (只需5处修改):
```python
def add_common_args(parser):
    parser.add_argument("--amp", type=bool, default=False, ...)  # 1处

def train_one_snr(..., use_amp=False):
    # 2-5处: 添加AMP逻辑
    if use_amp:
        scaler = GradScaler()
        with autocast():
            # ...
```

**总修改**: 1个文件 × 5处 = **5处修改**

**效率提升**: 66.7%

---

## 场景6：更改GPU优化默认值

### ❌ 重构前

**需求**: 将 `num_workers` 从4改为8

需要在3个文件中分别查找并修改：
```python
parser.add_argument("--num-workers", type=int, default=4, ...)
```

**风险**: 可能遗漏某些文件

---

### ✅ 重构后

**common.py** (只需1处):
```python
def add_common_args(parser):
    parser.add_argument("--num-workers", type=int, default=8, ...)
```

**优势**:
- ✅ 修改1次
- ✅ 所有实验自动更新
- ✅ 不会遗漏

---

## 📊 总体效率提升统计

| 场景 | 重构前修改次数 | 重构后修改次数 | 效率提升 |
|------|---------------|---------------|----------|
| 修改默认参数 | 3 | 1 | **66.7%** |
| 添加新参数 | 9 | 2 | **77.8%** |
| 修复Bug | 3 | 1 | **66.7%** |
| 添加新策略 | 180分钟 | 60分钟 | **66.7%** |
| 启用AMP | 15 | 5 | **66.7%** |
| 更改默认值 | 3 | 1 | **66.7%** |

**平均效率提升**: **~70%**

---

## 🎯 结论

通过创建 `common.py` 共享模块，成功实现了:

1. ✅ **"改一个就是改全部"** 的目标
2. ✅ **维护成本降低约70%**
3. ✅ **代码一致性得到保证**
4. ✅ **Bug修复效率大幅提升**
5. ✅ **新功能添加变得简单**

这种代码组织方式特别适合:
- 有多个相似实验的项目
- 需要频繁调整参数的研究代码
- 团队协作的项目
- 需要长期维护的代码库
