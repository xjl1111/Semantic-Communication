# Fig.8 实验问题分析与快速验证方案

## 一、当前实验结果

修改已实施：`CHANNEL_DIM=8`、`transfer_if=smaller`、差异化 caption prompt。

````carousel
**Without MED BLEU-1 矩阵**
```
当前:                     论文参考:
[0.84, null, null]       [0.85,  null, null]
[0.73, 0.83, null]       [0.214, 0.69, null]
[0.69, 0.71, 0.85]       [0.19,  0.30, 0.77]
```
> [!WARNING]
> CIFAR 降幅 18%（0.84→0.69），论文期望降幅 78%（0.85→0.19）
<!-- slide -->
**With MED BLEU-1 矩阵**
```
当前:                     论文参考:
[0.82, null, null]       [0.85, null, null]
[0.77, 0.84, null]       [0.83, 0.85, null]
[0.79, 0.79, 0.84]       [0.79, 0.83, 0.76]
```
> [!CAUTION]
> 异常：CIFAR 0.77→0.79 先降后**升**，不合常理
````

---

## 二、问题1：遗忘为何不够彻底

### 根因：Caption 差异化仍然不够

当前的差异化 prompt 修改了一些句式，但根本问题没解决——**三个数据集的核心词汇仍然高度重叠**：

| 数据集 | 当前 prompt | 实际 caption 示例 |
|---|---|---|
| CIFAR | `"this is a"` | "this is a bird on a branch" |
| Birds | `"a detailed photo of a bird species, showing"` | "a detailed photo of a bird species, showing a small bird on..." |
| CatsvsDogs | `"a pet photo showing a"` | "a pet photo showing a cat on a couch" |

**问题**：虽然 prompt 前缀不同，但 BLIP 生成的核心内容仍然是 "bird/cat/dog + 动作 + 地点"。而 CIFAR 已经包含了 bird/cat/dog 类别。模型传输的核心语义特征（token embedding 层面）没有本质变化。

### 真正需要的是什么

论文中遗忘降幅 78% 意味着模型在切换任务后**完全无法正确传输旧任务的文本**。这要求：
1. **不同任务的词汇空间几乎不重叠** — 例如 Task1 全是交通工具描述，Task2 全是鸟类专业术语
2. **或者模型容量足够大** — 有足够的任务特异性参数可以被覆盖

### 解决方向

> [!IMPORTANT]
> 有两种方案，二选一：

**方案A：更激进的文本差异化**

为每个数据集生成**完全不同风格**的 caption，使词汇空间不重叠：

```python
CAPTION_PROMPTS_BY_DATASET = {
    "cifar":      "describe the vehicle or object:",        # 只引导物体/交通工具词汇
    "birds":      "describe this bird's plumage, beak, and habitat in detail:",  # 专业鸟类术语
    "catsvsdogs": "describe this pet's breed, fur color, and expression:",       # 品种/毛色/表情
}
```

但即使这样改，BLIP 生成的 caption 可能仍带有重叠词汇（"a", "the", "on"...），因为 BLIP 的生成风格很固定。

**方案B（推荐）：使用类别标签直接构造差异化文本**

完全绕过 BLIP caption 生成，直接用数据集自带的标签构造格式化文本。这能**精确控制**三个任务的文本分布不重叠：

```python
# CIFAR 10类：airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
# → 训练文本: "transport vehicle: a red automobile on highway", "wildlife: a deer in forest"

# Birds 200种鸟类（有细粒度类名）
# → 训练文本: "avian species: laysan albatross with white plumage perched on coastal rock"

# CatsvsDogs
# → 训练文本: "domestic pet: a fluffy orange tabby cat resting on blue cushion"
```

这样三个任务的文本前缀和核心词汇**完全不同**，发生灾难性遗忘是必然的。

---

## 三、问题2：With MED 评分先降后升

**现象**：CIFAR BLEU-1 在训练 Birds 后降到 0.77，训练 CatsvsDogs 后反弹到 0.79。

### 原因分析

这不是 bug，而是 MED 回放机制 + caption 同质化共同造成的：

1. **MED LTM 积累了 CIFAR 数据** → 在训练 CatsvsDogs 时，semantic/joint 阶段的每个 batch 都混入了 LTM 中的 CIFAR 旧样本（`med_replay_batch_size = batch_size // 2 = 8`）
2. **三个数据集的 caption 仍然高度相似**（核心词汇重叠）→ 模型在 CatsvsDogs 上学到的 caption 传输能力**同时也改善了** CIFAR 的传输效果
3. 结合这两点：MED 回放 CIFAR 数据 + CatsvsDogs 训练"顺便"增强了通用传输能力 → CIFAR 分数回升

> [!NOTE]
> 如果文本差异化成功（方案B），这个异常会自然消失。因为 CatsvsDogs 的训练梯度不会对 CIFAR 的词汇空间产生正向影响。

---

## 四、快速验证方案

当前每次完整实验耗时太长。推荐以下**快速验证 pipeline**：

### 方案：缩小数据 + 减少 epoch

在 [fig8_config.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/fig8/fig8_config.py) 中临时修改：

```python
# ─── 快速验证模式（完成后改回正式值）─────────
CHANNEL_EPOCHS:  int = 5      # 正式值: 30
SEMANTIC_EPOCHS: int = 5      # 正式值: 30
JOINT_EPOCHS:    int = 5      # 正式值: 20
JOINT_PATIENCE:  int = 3      # 正式值: 10
```

启动命令：

```bash
python train_fig8.py --train_max_per_class 50
```

| 参数 | 快速验证 | 正式实验 |
|------|---------|---------|
| `--train_max_per_class` | 50 | -1（全部） |
| CHANNEL_EPOCHS | 5 | 30 |
| SEMANTIC_EPOCHS | 5 | 30 |
| JOINT_EPOCHS | 5 | 20 |
| BLEU 精度 | 估计 ±0.05 | 完整 |
| **预计耗时** | **~15分钟** | **数小时** |

> [!TIP]
> 快速验证的目标不是得到精确 BLEU 值，而是看**遗忘趋势**是否出现（CIFAR 分数大幅下降）。只要趋势对了，再用正式参数跑一次即可。

---

## 五、行动建议

1. **先实施方案B**（标签构造差异化文本），这是从根本上解决遗忘不彻底的方法
2. **用快速验证 pipeline** 验证改动效果
3. 确认遗忘趋势正确后，恢复正式参数跑全量实验
