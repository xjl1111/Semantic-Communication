# Fig.8 实验问题复盘与修订方案（Reviewed / No Prompt 版）

## 一、文档目的

本文件是在原 `implementation_plan.md` 基础上，结合论文 Fig.8 的设定与当前实验现象做出的**审阅后修订版**。

本版的核心修订原则是：

1. **取消 dataset-specific prompt**，正式主线改为 **无 prompt**。
2. **不使用标签构造文本**，保持 sender-side 仍然是 **BLIP caption**。
3. **不引入额外分类器或语义先验去修正 caption**。
4. Fig.8 的主变量仍然只保留：
   - `with MED`
   - `without MED`
5. 其它设置在两组实验之间必须严格一致，确保这是一个干净的 MED 消融。

---

## 二、当前实验现象（沿用已有观测）

当前已有修改包括：

- `CHANNEL_DIM = 8`
- `transfer_if = smaller`
- 之前尝试过差异化 caption prompt

当前观测矩阵如下：

### 1. Without MED / BLEU-1

```text
当前:                     论文参考:
[0.84, null, null]       [0.85,  null, null]
[0.73, 0.83, null]       [0.214, 0.69, null]
[0.69, 0.71, 0.85]       [0.19,  0.30, 0.77]
```

现象：
- CIFAR 从 `0.84 -> 0.69`，确实下降了；
- 但下降幅度明显小于论文图中 without MED 的粗略参考趋势；
- 说明当前实现的 **遗忘存在，但不够“彻底”**。

### 2. With MED / BLEU-1

```text
当前:                     论文参考:
[0.82, null, null]       [0.85, null, null]
[0.77, 0.84, null]       [0.83, 0.85, null]
[0.79, 0.79, 0.84]       [0.79, 0.83, 0.76]
```

现象：
- with MED 的整体趋势方向是对的；
- 但 `CIFAR: 0.77 -> 0.79` 出现轻微回升；
- 这个现象 **不必然是 bug**，也可能来自 MED 回放与后续训练共同作用。

---

## 三、对原方案的审阅结论

## 3.1 原方案中我保留的部分

以下内容我认为仍然合理：

1. **趋势判断基本成立**
   - 当前结果体现了 with/without MED 的差异；
   - 只是和论文图八相比，数值形态还不够像。

2. **“with MED 先降后微升不一定是 bug”**
   - 这个判断可以保留；
   - 在持续学习 + 记忆回放场景下，小幅回升可以先视为现象，而不是先判程序错。

3. **快速验证 pipeline 是合理的**
   - 小数据 + 少 epoch 先看趋势；
   - 趋势对了再跑正式实验。

## 3.2 原方案中我删掉或明确反对的部分

### A. 删除“继续使用 dataset-specific prompt 差异化”的主线建议

原因：
- 这会把发送端文本分布的人为差异和 MED 的作用混在一起；
- 对 Fig.8 来说，正式主线应尽量保持 sender-side 生成机制一致；
- 你当前已经决定改成 **无 prompt**，这比继续做按数据集差异化 prompt 更干净。

### B. 明确反对“使用标签直接构造差异化文本”

原因：
- 这会绕过 BLIP caption；
- sender-side 就不再是论文中的 BLIP-based KB；
- 这不是小修小补，而是直接换了 sender 机制；
- 因此不能作为 Fig.8 论文对齐主线。

### C. 不接受“必须让任务词汇空间强烈不重叠”作为核心论文依据

原因：
- 这是对现象的一种解释，不是论文明确给出的必要条件；
- 可以作为猜测，但不能当成主结论；
- 正式主线不应围绕这个猜测继续做更激进的人为文本改造。

---

## 四、修订后的主假设

本版采用的主假设如下：

1. Fig.8 主要要验证的是：
   - **在相同 sender-side caption 机制下**，
   - **有无 MED** 时，
   - 持续学习后旧任务 BLEU 的保持能力是否不同。

2. sender-side caption 应尽量保持“自然、统一、无任务特定先验”。

3. 因此，**无 prompt** 比 dataset-specific prompt 更适合作为 Fig.8 正式主线。

4. 如果在无 prompt 条件下：
   - with MED 仍显著优于 without MED，
   - 那就说明 MED 的作用更可信；
   - 即使绝对 BLEU 下降，也不影响消融的公正性。

---

## 五、修订后的正式方案（主线）

## 5.1 Caption 策略

正式主线统一改为：

```text
No prompt
```

即：
- 不使用 dataset-specific prompt；
- 不使用全局固定 prompt；
- 直接让 BLIP 按默认 captioning 生成文本。

### 这样做的理由

1. 避免人为引入不同数据集的文本分布差异；
2. 更接近“BLIP 自己从图像生成描述”的原始行为；
3. 让 Fig.8 更聚焦于 MED 本身，而不是 prompt 工程。

## 5.2 with / without MED 的比较要求

两组实验必须保证以下设置完全一致：

- 相同数据集顺序：`CIFAR -> Birds -> CatsVsDogs`
- 相同 BLIP 模型与推理参数
- 相同 caption 生成策略：**No prompt**
- 相同 `CHANNEL_DIM`
- 相同信道类型：`Rayleigh`
- 相同 epoch 配置
- 相同 batch size
- 相同 seed 策略（或相同 seed 列表）

唯一变量：

```text
MED = on / off
```

---

## 六、对当前现象的修订解释

## 6.1 为什么 without MED 遗忘还不够强

当前只能说：

- 你的实现已经出现遗忘；
- 但遗忘幅度小于论文图面的 rough trend；
- 原因可能包括：
  - bottleneck 不够强或任务不够难；
  - 训练阶段的保持能力仍偏强；
  - caption 底座质量较稳；
  - 其它工程实现细节（epoch、噪声、回放、优化器等）共同作用。

**本版不再把“prompt 差异化不够”作为首要根因。**

## 6.2 为什么 with MED 会轻微回升

当前可以接受的解释是：

- MED 在后续阶段持续回放旧样本；
- 模型在继续训练中对旧任务没有完全失去适配；
- 因此出现小幅回升，并不必然说明程序有错。

但如果回升过大，仍然需要排查：
- MED replay 配比是否过高；
- 评估矩阵是否读取错 checkpoint；
- 是否存在 train/val 泄漏；
- 是否存在旧数据在后阶段被重复训练过多。

---

## 七、快速验证方案（保留并微调）

快速验证目标：

- 不追求最终精确数值；
- 只看在 **No prompt** 条件下，
  - with MED / without MED 的差异是否仍然存在；
  - without MED 是否会比 with MED 更明显遗忘旧任务。

### 建议配置

在 `fig8_config.py` 中临时修改：

```python
CHANNEL_EPOCHS  = 5
SEMANTIC_EPOCHS = 5
JOINT_EPOCHS    = 5
JOINT_PATIENCE  = 3
```

启动命令：

```bash
python train_fig8.py --train_max_per_class 50
```

并确保：

```text
caption_mode = blip
prompt = None
```

### 快速验证关注点

只看下面三件事：

1. `without MED` 的 Cifar 是否在后续任务后明显下降；
2. `with MED` 的 Cifar 是否下降更少；
3. with/without MED 的趋势差异是否仍存在。

---

## 八、正式实验建议

当快速验证通过后，再恢复正式参数：

```python
CHANNEL_EPOCHS  = 30
SEMANTIC_EPOCHS = 30
JOINT_EPOCHS    = 20
JOINT_PATIENCE  = 10
```

正式实验中：

1. 不再使用 dataset-specific prompt；
2. 不使用标签构造文本；
3. 不使用基于外部分类器的 rerank；
4. 不使用“没有动物词就重试”这类数据集特定规则；
5. 只保留通用失败恢复（如空输出、重复塌缩、纯标点）作为可选防护。

---

## 九、修订后的行动建议

### 主线执行顺序

1. **删除所有 dataset-specific prompt 配置**
2. **将 Fig.8 caption 统一改为 No prompt**
3. **保持 BLIP caption 生成，不使用标签模板**
4. **先跑快速验证版**
5. **观察 with/without MED 趋势是否拉开**
6. **若趋势成立，再跑正式全量实验**

### 不再推荐的操作

1. 不再继续调 dataset-specific prompt
2. 不再尝试“标签构造文本”替代 BLIP caption
3. 不再把“词汇空间不重叠”当作主修复方向
4. 不在正式主线中加入任何类别先验规则

---

## 十、最终决策

本次审阅后的正式建议是：

> **Fig.8 主线实验改为 No prompt。**
>
> sender-side 继续使用 BLIP 默认 captioning；
> 不使用按数据集区分的 prompt；
> 不使用标签构造文本；
> 让 with MED / without MED 的差异尽可能只来自 MED 本身。

