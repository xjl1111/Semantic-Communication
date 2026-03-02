# 归一化层说明 - Encoder输入归一化

## 问题

用户正确指出：论文架构图（Fig.2）中Encoder的**第一层**是Normalization层，用于将像素从[0,255]归一化到[0,1]，但代码实现中没有明确体现。

## 论文原文

> "At the encoder, **the normalization layer is followed by five convolutional layers**. Since the statistics of the input data are generally not known at the decoder, **the input images are normalized by the maximum pixel value 255, producing pixel values in the [0, 1] range.**"

## Encoder的两个Normalization层

根据论文Fig.2，Encoder有**两个**Normalization层：

1. **输入归一化（开头）**: 将像素值从 [0, 255] → [0, 1]
   - 位置：5层卷积**之前**
   - 作用：像素值归一化
   - 公式：`x_norm = x / 255`

2. **功率归一化（结尾）**: 约束信道输入功率 E[|z|²] = 1
   - 位置：5层卷积**之后**
   - 作用：满足功率约束
   - 公式：`z = sqrt(kP) * z_tilde / ||z_tilde||`

## 实现方式

### 当前实现

```python
class Encoder(nn.Module):
    def __init__(self, ..., input_normalized: bool = True):
        # input_normalized参数控制是否需要输入归一化
        self.input_normalized = input_normalized
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入归一化（论文Fig.2第一层）
        if not self.input_normalized:
            x = x / 255.0  # [0,255] → [0,1]
        
        # 5层卷积+PReLU
        x = self.conv1(x)
        ...
        x = self.conv5(x)
        
        # 功率归一化（论文Fig.2最后一层）
        energy = x.pow(2).sum(...)
        x = x * scale
        
        return x
```

### 两种使用方式

#### 方式1：数据预处理时归一化（默认）
```python
# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),  # 自动归一化到[0,1]
])

# Encoder配置
encoder = Encoder(c=8, input_normalized=True)  # 默认值
```

#### 方式2：Encoder内部归一化
```python
# 数据加载（不归一化）
transform = transforms.Compose([
    transforms.PILToTensor(),  # 转tensor但不归一化，保持[0,255]
])

# Encoder配置
encoder = Encoder(c=8, input_normalized=False)  # Encoder内部会/255
```

## 为什么使用方式1？

当前所有实验使用**方式1**（`input_normalized=True`）的原因：

1. **PyTorch标准实践**: `transforms.ToTensor()` 是标准做法
2. **代码简洁**: 避免在Encoder forward中重复归一化
3. **性能**: 归一化只做一次（数据加载时），不是每次forward
4. **兼容性**: 与现有PyTorch数据流水线无缝集成

但**严格按照论文架构图**，应该是方式2（Encoder内部归一化）。

## 当前代码的正确性

两种方式**在数学上完全等价**：
- 方式1: ToTensor归一化 → Encoder (skip归一化) → 5层conv → 功率归一化
- 方式2: 不归一化 → Encoder (像素/255) → 5层conv → 功率归一化

**结果完全相同**，因为归一化是线性操作，只是在不同位置执行。

## 对实验的影响

✅ **无影响** - 所有实验结果保持不变，因为：
1. 当前使用`input_normalized=True`（默认值）
2. 数据通过`ToTensor()`已归一化
3. Encoder的forward检测到`input_normalized=True`，跳过/255操作
4. 数学上等价于论文架构

## 结论

✅ **代码实现正确** - 输入归一化已完成（在数据加载时）  
✅ **架构符合论文** - 添加了`input_normalized`参数以支持两种方式  
✅ **灵活性增强** - 用户可选择在哪里归一化  
✅ **测试全部通过** - 功能验证无误  

---

**修正日期**: 2026年1月25日  
**修正原因**: 用户指出论文明确提到输入归一化，代码应体现架构图的完整性  
**修正方式**: 添加`input_normalized`参数和条件归一化逻辑  
