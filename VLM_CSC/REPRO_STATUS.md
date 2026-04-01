# VLM-CSC 严格复现 — 三类汇报 (REPRO_STATUS.md)

> 本文档将复现工作中的每个技术决策点分为三类：
> - ✅ **论文明确 (Paper-Explicit)** — 论文给出了精确描述，代码严格遵循
> - 🟡 **工程近似 (Engineering Approximation)** — 论文描述不够精确，代码使用合理推测
> - 🔴 **严格阻断 (Hard Blocker)** — 论文信息不足，无法完成，已在代码中阻断

---

## 一、论文明确 (Paper-Explicit) ✅

| # | 条目 | 论文依据 | 代码位置 | 状态 |
|---|------|----------|----------|------|
| 1 | NAM 结构: 4层 FF (56,128,56,56) | §III-B eq. | `communication_modules.py::NAM` | ✅ 已锁定 |
| 2 | feature_dim = 128 | §III-A | `paper_repro_lock.py` | ✅ 已锁定 |
| 3 | Semantic Encoder/Decoder: 3层 Transformer, 8头 | §III-A | `paper_repro_lock.py` | ✅ 已锁定 |
| 4 | Channel Encoder/Decoder: FF [256,128] | §III-A | `paper_repro_lock.py` | ✅ 已锁定 |
| 5 | STM max=500 | §III-D MED | `paper_repro_lock.py` | ✅ 已锁定 |
| 6 | MED τ=10, λ=0.05, RBF kernel | §III-D | `paper_repro_lock.py` | ✅ 已锁定 |
| 7 | Receiver KB = Stable Diffusion | §III-A | `paper_repro_lock.py` | ✅ 已锁定 |
| 8 | Sender KB: BLIP + RAM 双 sender | §III-A | `fig7_config.py` | ✅ 强制 |
| 9 | Fig7: AWGN + CatsvsDogs + SSQ | §IV-A | `fig7_config.py` + `paper_repro_lock.py` | ✅ 已锁定 |
| 10 | Fig8: Rayleigh + 持续学习 + BLEU | §IV-B | `fig8_config.py` + `paper_repro_lock.py` | ✅ 已锁定 |
| 11 | Fig9: AWGN NAM 消融 + BLEU | §IV-C | `fig9_config.py` + `paper_repro_lock.py` | ✅ 已锁定 |
| 12 | Fig9 with-NAM: uniform_range [0,10] dB | §IV-C | `paper_repro_lock.py` | ✅ 已锁定 |
| 13 | Fig9 without-NAM: fixed {0,2,4,8} dB | §IV-C | `paper_repro_lock.py` | ✅ 已锁定 |
| 14 | SSQ = ST(reconstructed)/ST(original) | §IV-A eq.(19) | `eval_metrics.py` | ✅ |
| 15 | 三阶段训练: channel → semantic → joint | §III-E | `train_experiment.py` | ✅ 已实现 |
| 16 | Channel phase: 冻结 CKB+Semantic, 训练 Channel+NAM | §III-E | `train_phase_utils.py` | ✅ |
| 17 | Semantic phase: 冻结 CKB+Channel, 训练 Semantic+NAM | §III-E | `train_phase_utils.py` | ✅ |
| 18 | use_nam=False 时 NAM 结构消除（零参数） | §IV-C 消融 | `communication_modules.py` | ✅ |
| 19 | CKB (BLIP/RAM/SD) 全程冻结不可训练 | §III-A | `train_phase_utils.py` | ✅ |
| 20 | CLIP zero-shot 作为分类后端 | §IV-A | `target.py` | ✅ |
| 21 | Fig8 数据集序列: cifar→birds→catsvsdogs | §IV-B | `paper_repro_lock.py` | ✅ |
| 22 | Fig8 MED toggle: with_med vs without_med | §IV-B | `paper_repro_lock.py` | ✅ |

---

## 二、工程近似 (Engineering Approximation) 🟡

| # | 条目 | 论文描述 | 实际实现 | 差异说明 |
|---|------|----------|----------|----------|
| 1 | Channel phase 损失函数 | 互信息(MI)最小化 | `masked_sequence_mse` | 论文未给出精确 MI loss 公式 |
| 2 | Compression ratio 定义 | Fig.10 metric | 维度比 src/tx | 论文未精确定义计算方式 |
| 3 | MED 记忆回放超参数 | STM-based replay | batch=4, ratio=0.5, weight=1.0 | 论文未给出回放细节 |
| 4 | Joint phase α/β 权重 | §III-E | 可配置 (默认 0.5/0.5) | 论文未给出精确权重值 |
| 5 | 训练 epoch 数 | §IV 实验设置 | 可配置 | 论文未明确给出 epoch 数 |
| 6 | 学习率及 scheduler | §IV | lr=1e-4, AdamW | 论文未完全明确 |
| 7 | RAM CKB 输出格式 | "This image contains: ..." | 逗号分隔标签 | 基于 RAM 原始输出格式推测 |

---

## 三、严格阻断 (Hard Blocker) 🔴

| # | 条目 | 缺失原因 | 影响范围 | 阻断位置 |
|---|------|----------|----------|----------|
| 1 | JSCC 基线 | 论文未提供实现代码/权重 | Fig.10 | `eval_experiment.py` RuntimeError |
| 2 | WITT 基线 | 论文未提供 TorchScript 模型 | Fig.10 | `eval_experiment.py` RuntimeError |

---

## 四、审计基础设施

| 模块 | 功能 | 位置 |
|------|------|------|
| `formal_guard.py` | 正式实验准入门卫 | `exp/audit/` |
| `checkpoint_meta.py` | 检查点元数据 + git hash | `exp/audit/` |
| `anti_cheat_checks.py` | 13 项反作弊清单 | `exp/audit/` |
| `smoke_formal_isolation.py` | smoke/formal 目录隔离 | `exp/audit/` |
| `paper_repro_lock.py` | 协议参数锁 + NAM 断言 | `exp/` |

---

## 五、当前可正式运行的实验

| Figure | 可否正式运行 | 阻断原因 |
|--------|-------------|----------|
| Fig.7 | ✅ 可以 | — |
| Fig.8 | ✅ 可以 | — |
| Fig.9 | ✅ 可以 | — |
| Fig.10 | ⚠️ 仅 vlm_csc | JSCC/WITT 基线缺失 |

---

*最后更新: 2024 — 由审计模块自动生成*
