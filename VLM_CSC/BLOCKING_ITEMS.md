# VLM-CSC 严格复现 — 阻断项清单 (BLOCKING_ITEMS.md)

> 本文档记录严格复现论文 "Visual Language Model-Based Cross-Modal Semantic Communication Systems" 过程中
> 遇到的阻断项。阻断项 = 论文未提供足够信息、无法仅凭论文内容完成的复现环节。

---

## 1. Channel Phase 损失函数 (MI Loss)

| 字段 | 内容 |
|------|------|
| **论文描述** | §III-C 指出 channel phase 目标为最小化互信息 (mutual information) |
| **缺失信息** | 论文未给出精确的 MI loss 数学公式 |
| **当前实现** | `masked_sequence_mse` — 重建序列与教师序列的 masked MSE |
| **影响** | channel phase 训练效果可能偏离论文原始结果 |
| **分类** | 🟡 工程近似 (Engineering Approximation) |
| **代码位置** | `exp/train_phase_utils.py::masked_sequence_mse()` |

---

## 2. JSCC 基线实现

| 字段 | 内容 |
|------|------|
| **论文描述** | Fig.10 要求对比 Deep JSCC 基线 |
| **缺失信息** | 论文未提供 JSCC 实现代码或预训练权重 |
| **当前状态** | `deep_jscc/` 为本项目自行编写，无法确认与论文所用一致 |
| **影响** | Fig.10 基线对比不可作为正式结果 |
| **分类** | 🔴 严格阻断 (Hard Blocker) |
| **代码位置** | `exp/eval_experiment.py::_run_baseline_performance()` — RuntimeError 阻断 |

---

## 3. WITT 基线实现

| 字段 | 内容 |
|------|------|
| **论文描述** | Fig.10 要求对比 WITT 基线 |
| **缺失信息** | 论文未提供 WITT TorchScript 模型或训练代码 |
| **当前状态** | 代码假设 `torch.jit.load` 加载预训练模型，但模型不存在 |
| **影响** | Fig.10 基线对比不可作为正式结果 |
| **分类** | 🔴 严格阻断 (Hard Blocker) |
| **代码位置** | `exp/eval_experiment.py::_run_baseline_performance()` — RuntimeError 阻断 |

---

## 4. Compression Ratio 精确定义

| 字段 | 内容 |
|------|------|
| **论文描述** | Fig.10 metrics 包含 compression_ratio |
| **缺失信息** | 论文未精确定义 compression ratio = src_bits / tx_bits 还是 src_dims / tx_dims 等 |
| **当前实现** | `target.py::compute_compression_ratio()` — 基于维度比 |
| **影响** | 数值可能与论文不完全一致 |
| **分类** | 🟡 工程近似 (Engineering Approximation) |

---

## 5. MED 记忆回放细节

| 字段 | 内容 |
|------|------|
| **论文描述** | §III-D 描述了 MED 模块的 STM/RBF kernel 选择机制 |
| **缺失信息** | 训练阶段记忆回放的具体 batch 构成、回放频率、回放权重等细节 |
| **当前实现** | `med_replay_batch_size=4, med_replay_stm_ratio=0.5, med_replay_weight=1.0` |
| **影响** | Fig.8 持续学习实验的记忆遗忘曲线可能有偏差 |
| **分类** | 🟡 工程近似 (Engineering Approximation) |

---

## 摘要

| 分类 | 数量 | 说明 |
|------|------|------|
| 🔴 严格阻断 (Hard Blocker) | 2 | JSCC/WITT 基线 — 需要原作者提供 |
| 🟡 工程近似 (Engineering Approximation) | 3 | MI loss, compression ratio, MED replay |
| ✅ 论文明确 (Paper-Explicit) | — | 见 REPRO_STATUS.md |
