# VLM_CSC exp 目录模块化重构 — 完整详细计划

## 1. 背景与目标

[exp/](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/experiment_bootstrap.py#88-107) 目录下核心文件过于庞大：
- [train_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py) — **1800 行**
- [eval_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py) — **1032 行**
- [common_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py) — **347 行**（包含了训练/评估/诊断全部共用的函数，需要拆分归位）

**目标**：按职责拆分到 [train/](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#971-1112) 和 [eval/](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#1013-1016) 子包 + `common/` 子包，每个文件 **≤ 500 行**，公开 API 不变。

---

## 2. common_experiment.py 依赖分析

[common_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py) 的 347 行代码被以下 **8 个文件** 引用：

| 调用者 | 使用的函数/类 |
|---|---|
| [train_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py) | [TaskDatasetManager](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#75-245), [assert_fig8_variant_model_state](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#326-347), [build_vlm_system](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#256-313), [chunk_records](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#247-254), [collect_binary_images_from_split](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#49-61), [collect_generic_images_from_split](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#63-73), [configure_runtime_logging](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#18-37), [load_module_from_file](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#39-47), [resolve_fig8_variant_med_config](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#315-324) |
| [eval_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py) | [TaskDatasetManager](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#75-245), [assert_fig8_variant_model_state](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#326-347), [build_vlm_system](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#256-313), [chunk_records](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#247-254), [collect_binary_images_from_split](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#49-61), [configure_runtime_logging](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#18-37), [load_module_from_file](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#39-47), [resolve_fig8_variant_med_config](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#315-324) |
| [eval_metrics.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_metrics.py) | [chunk_records](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#247-254) |
| [diag_snr_text.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/diag_snr_text.py) | [load_module_from_file](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#39-47), [build_vlm_system](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#256-313) |
| [diag_e2e_text.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/diag_e2e_text.py) | [load_module_from_file](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#39-47) |
| [diag_decoder_sensitivity.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/diag_decoder_sensitivity.py) | [load_module_from_file](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#39-47) |
| [diag_channel_capacity.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/diag_channel_capacity.py) | [load_module_from_file](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#39-47) |
| [tools/rebuild_caption_cache.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/tools/rebuild_caption_cache.py) | [TaskDatasetManager](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#75-245), [build_vlm_system](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#256-313), [collect_binary_images_from_split](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#49-61), [load_module_from_file](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#39-47) |

**按归属分类**：

| 类别 | 函数/类 | 行数 | 归属 |
|---|---|---|---|
| **通用工具** | `LABEL_MAP`, [configure_runtime_logging](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#18-37), [load_module_from_file](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#39-47), [collect_binary_images_from_split](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#49-61), [collect_generic_images_from_split](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#63-73), [chunk_records](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#247-254) | ~75行 | → `common/utils.py` |
| **数据集管理** | [TaskDatasetManager](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#75-245) | ~170行 | → `common/dataset_manager.py` |
| **模型构建** | [build_vlm_system](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#256-313) | ~60行 | → `common/model_builder.py` |
| **Fig8 变体** | [resolve_fig8_variant_med_config](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#315-324), [assert_fig8_variant_model_state](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#326-347) | ~35行 | → `common/fig8_variant.py` |

---

## 3. 最终目录结构

```
VLM_CSC/exp/
│
├── common/                             # 共享子包 [NEW]
│   ├── __init__.py                     # 公开全部 API
│   ├── utils.py                        # LABEL_MAP, configure_runtime_logging,
│   │                                   # load_module_from_file, collect_*_images,
│   │                                   # chunk_records
│   ├── dataset_manager.py              # TaskDatasetManager
│   ├── model_builder.py                # build_vlm_system
│   └── fig8_variant.py                 # resolve_fig8_variant_med_config,
│                                       # assert_fig8_variant_model_state
│
├── train/                              # 训练子包 [NEW]
│   ├── __init__.py                     # 公开 API: TrainConfig, run_training 等
│   ├── config.py                       # TrainConfig 数据类 + _validate_train_phase_config
│   ├── phases.py                       # run_channel_phase, run_semantic_phase, run_joint_phase
│   ├── protocol.py                     # run_paper_training_protocol, train_sender
│   ├── fig8_continual.py               # run_fig8_continual_training
│   ├── helpers.py                      # SNR 采样、checkpoint 加载/元数据注入、MED 批次、
│   │                                   # BLEU 计算、CSV 写入等内部工具
│   └── router.py                       # _run_training_core + run_fig7/8/9/10_protocol
│                                       # + run_training
│
├── eval/                               # 评估子包 [NEW]
│   ├── __init__.py                     # 公开 API: EvalConfig, run_evaluation 等
│   ├── config.py                       # EvalConfig 数据类
│   ├── core.py                         # _run_evaluation_core（通用 SNR 循环）
│   ├── fig8_continual.py               # _run_continual_bleu_map
│   ├── baselines.py                    # _run_baseline_performance, _build_jscc_pipeline
│   ├── validators.py                   # check_sd_assets, fig8 严格协议校验, checkpoint 查找
│   └── router.py                       # run_fig7/8/9/10_eval + run_evaluation
│
├── common_experiment.py                # [兼容层] → from common import *
├── train_experiment.py                 # [兼容层] → from train import *
├── eval_experiment.py                  # [兼容层] → from eval import *
│
├── eval_metrics.py                     # [保留不动] 指标计算与可视化 (421行)
├── train_phase_utils.py                # [保留不动] 参数冻结/解冻 (183行)
├── shared_config.py                    # [保留不动] 共享配置构建 (175行)
├── caption_cache.py                    # [保留不动]
├── resume_manager.py                   # [保留不动]
├── experiment_bootstrap.py             # [保留不动]
├── target.py                           # [保留不动]
├── paper_repro_lock.py                 # [保留不动]
├── diag_*.py                           # [保留不动] 诊断脚本
├── _*.py                               # [保留不动] 快速测试脚本
├── tools/                              # [保留不动]
├── audit/                              # [保留不动]
├── results/                            # [保留不动]
└── fig7/ fig8/ fig9/ fig10/            # [保留不动] 各实验入口
```

---

## 4. 逐文件拆分明细

### 4.1 `common/` 子包

---

#### [NEW] `common/__init__.py` (~30行)

```python
"""VLM_CSC 实验共享工具包。"""
from common.utils import (
    LABEL_MAP,
    configure_runtime_logging,
    load_module_from_file,
    collect_binary_images_from_split,
    collect_generic_images_from_split,
    chunk_records,
)
from common.dataset_manager import TaskDatasetManager
from common.model_builder import build_vlm_system
from common.fig8_variant import (
    resolve_fig8_variant_med_config,
    assert_fig8_variant_model_state,
)

__all__ = [
    "LABEL_MAP", "configure_runtime_logging", "load_module_from_file",
    "collect_binary_images_from_split", "collect_generic_images_from_split",
    "chunk_records", "TaskDatasetManager", "build_vlm_system",
    "resolve_fig8_variant_med_config", "assert_fig8_variant_model_state",
]
```

---

#### [NEW] `common/utils.py` (~75行)

**来源**：[common_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py) L1-73

**内容**：
```python
# 搬移以下内容（原封不动）：
#   - LABEL_MAP 常量 (L15)
#   - configure_runtime_logging() (L18-36)
#   - load_module_from_file() (L39-46)
#   - collect_binary_images_from_split() (L49-60)
#   - collect_generic_images_from_split() (L63-72)
#   - chunk_records() (L247-253)   ← 注意这个在原文件较后面
```

---

#### [NEW] `common/dataset_manager.py` (~170行)

**来源**：[common_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py) L75-244

**内容**：
```python
# 完整搬移 TaskDatasetManager 类
# import 依赖：
#   - from common.utils import collect_binary_images_from_split, collect_generic_images_from_split
#   - 标准库: random, pathlib, typing
```

---

#### [NEW] `common/model_builder.py` (~60行)

**来源**：[common_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py) L256-312

**内容**：
```python
# 完整搬移 build_vlm_system() 函数
# import 依赖：
#   - contextlib, io (标准库)
```

---

#### [NEW] `common/fig8_variant.py` (~35行)

**来源**：[common_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py) L315-347

**内容**：
```python
# 完整搬移：
#   - resolve_fig8_variant_med_config() (L315-323)
#   - assert_fig8_variant_model_state() (L326-347)
```

---

#### [MODIFY] [common_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py) → 兼容层 (~5行)

```python
"""兼容层：保持旧 import 路径可用。"""
from common import *  # noqa: F401,F403
```

---

### 4.2 [train/](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#971-1112) 子包

---

#### [NEW] `train/__init__.py` (~30行)

```python
"""VLM_CSC 训练子包。"""
from train.config import TrainConfig
from train.phases import run_channel_phase, run_semantic_phase, run_joint_phase
from train.protocol import run_paper_training_protocol, train_sender
from train.fig8_continual import run_fig8_continual_training
from train.router import (
    run_training,
    run_fig7_protocol,
    run_fig8_protocol,
    run_fig9_protocol,
    run_fig10_protocol,
)
```

---

#### [NEW] `train/config.py` (~120行)

**来源**：[train_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py) L62-105 + L1114-1157

**内容**：
| 来源行号 | 搬移内容 |
|---|---|
| L62-105 | [TrainConfig](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#62-105) 数据类定义（全部字段 + 默认值） |
| L107-113 | [set_seed()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#107-114) |
| L116-117 | [load_vlm_module()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#116-118) |
| L1114-1157 | [_validate_train_phase_config()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#1114-1157) |

**import 依赖**：
- `from common.utils import load_module_from_file`
- `from train_phase_utils import require_phase_block as _require_phase_block`
- 标准库: `dataclasses`, `pathlib`, `typing`, `random`, `numpy`

---

#### [NEW] `train/helpers.py` (~300行)

**来源**：[train_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py) 中的内部辅助函数

**搬移清单**：

| 来源行号 | 函数名 | 说明 |
|---|---|---|
| L17-28 | `_TRAIN_CAT_PAT`, `_TRAIN_DOG_PAT`, [_train_text_matches_label()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#22-29) | 文本匹配正则 |
| L120-144 | [_snr_sampler()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#120-126), [_resolve_train_snr()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#128-145) | SNR 采样 |
| L147-164 | [_load_phase_best_checkpoint_strict()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#147-165) | 检查点加载 |
| L167-234 | [_sample_memory_batch_for_step()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#167-194), [_prepare_merged_batch()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#196-235) | MED 批次合并 |
| L237-266 | [_update_med_and_check()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#237-267) | MED 更新验证 |
| L269-291 | [_run_semantic_train_step_merged()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#269-279), [_run_joint_train_step_merged()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#281-292) | 步骤执行 |
| L294-336 | [_enrich_saved_checkpoints()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#294-337) | 检查点元数据注入 |
| L1159-1172 | [_compute_bleu_n()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#1159-1173) | BLEU 计算 |
| L1175-1209 | [_evaluate_bleu_on_records()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#1175-1210) | 记录级 BLEU 评估 |
| L1212-1229 | [_write_matrix_csv()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#1212-1230) | CSV 输出 |
| L1232-1268 | [_validate_fig8_variant_checkpoint_map_complete()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#1232-1269) | Fig8 checkpoint map 验证 |

**import 依赖**：
- `from train_phase_utils import assert_channel_forward_contract, assert_semantic_forward_contract, masked_sequence_mse, compute_info_nce_sequence`
- `from common.utils import chunk_records`
- 标准库: [re](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_phase_utils.py#6-9), [csv](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#1212-1230), `pathlib`, `typing`

---

#### [NEW] `train/phases.py` (~500行)

**来源**：[train_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py)

**搬移清单**：

| 来源行号 | 函数名 | 说明 | 行数 |
|---|---|---|---|
| L339-471 | [run_channel_phase()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#339-472) | 信道编解码器训练阶段 | ~133 |
| L474-616 | [run_semantic_phase()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#474-617) | 语义编解码器训练阶段 | ~143 |
| L619-829 | [run_joint_phase()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#619-830) | 联合训练阶段（交替信道/语义步骤） | ~211 |

**import 依赖**：
- `from train.helpers import _resolve_train_snr, _prepare_merged_batch, _update_med_and_check, _run_semantic_train_step_merged, _run_joint_train_step_merged, _train_text_matches_label`
- `from train_phase_utils import set_trainable_for_channel_phase, set_trainable_for_semantic_phase, set_trainable_for_joint_channel_step, set_trainable_for_joint_semantic_step, collect_trainable_params, assert_channel_forward_contract, masked_sequence_mse, compute_info_nce_sequence`
- `PIL.Image`, `tqdm`, `torch`

---

#### [NEW] `train/protocol.py` (~280行)

**来源**：[train_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py) L832-1112

**搬移清单**：

| 来源行号 | 函数名 | 说明 | 行数 |
|---|---|---|---|
| L832-968 | [run_paper_training_protocol()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#832-969) | 3阶段论文训练协议（channel→semantic→joint） | ~137 |
| L971-1112 | [train_sender()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#971-1112) | 单 sender 完整训练流程（构建模型→缓存→训练→元数据） | ~142 |

**import 依赖**：
- `from train.phases import run_channel_phase, run_semantic_phase, run_joint_phase`
- `from train.helpers import _load_phase_best_checkpoint_strict, _enrich_saved_checkpoints`
- `from common.model_builder import build_vlm_system`
- `from common.utils import chunk_records`
- `from caption_cache import ensure_captions_for_sender`
- `from resume_manager import ResumeManager`

---

#### [NEW] `train/fig8_continual.py` (~370行)

**来源**：[train_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py) L1271-1599

**搬移清单**：

| 来源行号 | 函数名 | 说明 |
|---|---|---|
| L1271-1599 | [run_fig8_continual_training()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#1271-1600) | Fig8 持续学习训练（variant×sender×task 三层循环，含 BLEU 监控） |

**import 依赖**：
- `from train.protocol import run_paper_training_protocol`
- `from train.helpers import _load_phase_best_checkpoint_strict, _evaluate_bleu_on_records, _write_matrix_csv, _validate_fig8_variant_checkpoint_map_complete`
- `from train.config import TrainConfig, load_vlm_module, _validate_train_phase_config`
- `from common import TaskDatasetManager, build_vlm_system, assert_fig8_variant_model_state, resolve_fig8_variant_med_config, chunk_records`
- `from caption_cache import ensure_captions_for_sender`
- `from resume_manager import ResumeManager, prompt_resume_or_restart`

---

#### [NEW] `train/router.py` (~200行)

**来源**：[train_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py) L1602-1800

**搬移清单**：

| 来源行号 | 函数名 | 说明 |
|---|---|---|
| L1602-1767 | [_run_training_core()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#1602-1767) | 训练核心入口（参数校验、fig8 分支、单数据集训练） |
| L1769-1773 | [run_fig7_protocol()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#1769-1774) | |
| L1776-1778 | [run_fig8_protocol()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#1776-1779) | |
| L1781-1783 | [run_fig9_protocol()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#1781-1784) | |
| L1786-1788 | [run_fig10_protocol()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#1786-1789) | |
| L1791-1799 | [run_training()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#1791-1800) | 总入口路由 |

**import 依赖**：
- `from train.config import TrainConfig, set_seed, load_vlm_module, _validate_train_phase_config`
- `from train.protocol import train_sender`
- `from train.fig8_continual import run_fig8_continual_training`
- `from common.utils import configure_runtime_logging, collect_binary_images_from_split, chunk_records`
- `from resume_manager import ResumeManager, prompt_resume_or_restart`

---

#### [MODIFY] [train_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py) → 兼容层 (~5行)

```python
"""兼容层：保持旧 import 路径可用。"""
from train import *  # noqa: F401,F403
```

---

### 4.3 [eval/](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#1013-1016) 子包

---

#### [NEW] `eval/__init__.py` (~20行)

```python
"""VLM_CSC 评估子包。"""
from eval.config import EvalConfig
from eval.router import (
    run_evaluation,
    run_fig7_eval,
    run_fig8_continual_evaluation,
    run_fig9_eval,
    run_fig10_baseline_evaluation,
)
```

---

#### [NEW] `eval/config.py` (~50行)

**来源**：[eval_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py) L34-84

**内容**：[EvalConfig](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#34-81) 数据类定义 + [load_module()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#83-85)

---

#### [NEW] `eval/validators.py` (~100行)

**来源**：[eval_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py)

**搬移清单**：

| 来源行号 | 函数名 | 说明 |
|---|---|---|
| L87-93 | [_file_sha256()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#87-93) | 文件 SHA256 |
| L95-111 | [check_sd_assets()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#95-112) | SD 模型完整性检查 |
| L114-141 | [_get_fig8_variant_checkpoint()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#114-141) | Fig8 checkpoint 查找 |
| L143-159 | [_normalize_fig8_med_variants()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#143-160) | Fig8 变体标准化 |
| L648-672 | [_validate_fig8_strict_protocol_inputs()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#648-672) | Fig8 严格协议校验 |

---

#### [NEW] `eval/fig8_continual.py` (~260行)

**来源**：[eval_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py) L162-398

**内容**：[_run_continual_bleu_map()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#162-399) 函数

**import 依赖**：
- `from eval.config import EvalConfig`
- `from eval.validators import _get_fig8_variant_checkpoint, _normalize_fig8_med_variants`
- `from common import TaskDatasetManager, build_vlm_system, assert_fig8_variant_model_state, resolve_fig8_variant_med_config`
- `from eval_metrics import evaluate_bleu_sender_snr, plot_matrix_heatmap`

---

#### [NEW] `eval/baselines.py` (~250行)

**来源**：[eval_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py) L401-645

**内容**：[_run_baseline_performance()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#431-646) + [_build_jscc_pipeline()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#401-429)

**import 依赖**：
- `from eval.config import EvalConfig, load_module`
- `from eval.validators import check_sd_assets`
- `from common import build_vlm_system, collect_binary_images_from_split`
- `from eval_metrics import evaluate_perf_sender_snr`

---

#### [NEW] `eval/core.py` (~330行)

**来源**：[eval_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py) L674-1000

**内容**：[_run_evaluation_core()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#674-1001) — 通用 SNR 循环 + CSV/plot 输出

**import 依赖**：
- `from eval.config import EvalConfig, load_module`
- `from eval.validators import check_sd_assets, _file_sha256, _validate_fig8_strict_protocol_inputs`
- `from eval.fig8_continual import _run_continual_bleu_map`
- `from eval.baselines import _run_baseline_performance`
- `from common import build_vlm_system, collect_binary_images_from_split, chunk_records, configure_runtime_logging`
- `from eval_metrics import evaluate_ssq_sender_snr, evaluate_bleu_sender_snr, evaluate_perf_sender_snr, plot_curve, save_visual_samples`

---

#### [NEW] `eval/router.py` (~40行)

**来源**：[eval_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py) L1003-1032

**搬移清单**：

| 来源行号 | 函数名 |
|---|---|
| L1003-1005 | [run_fig7_eval()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#1003-1006) |
| L1008-1010 | [run_fig8_continual_evaluation()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#1008-1011) |
| L1013-1015 | [run_fig9_eval()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#1013-1016) |
| L1018-1020 | [run_fig10_baseline_evaluation()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#1018-1021) |
| L1023-1032 | [run_evaluation()](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#1023-1032) |

---

#### [MODIFY] [eval_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py) → 兼容层 (~5行)

```python
"""兼容层：保持旧 import 路径可用。"""
from eval import *  # noqa: F401,F403
```

---

## 5. 不变文件

| 文件 | 行数 | 理由 |
|---|---|---|
| [eval_metrics.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_metrics.py) | 421 | 职责单一（只做指标计算和绘图），体量合理 |
| [train_phase_utils.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_phase_utils.py) | 183 | 只做参数冻结/解冻和合约检查，很小 |
| [shared_config.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/shared_config.py) | 175 | 配置构建函数，很小 |
| [caption_cache.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/caption_cache.py) | ~300 | 缓存管理，独立模块 |
| [resume_manager.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/resume_manager.py) | ~180 | 断点续传，独立模块 |
| [experiment_bootstrap.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/experiment_bootstrap.py) | 107 | 实验引导，独立模块 |
| [target.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/target.py) | ~250 | 下游任务指标 |
| [paper_repro_lock.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/paper_repro_lock.py) | ~260 | 复现锁定 |
| `diag_*.py` | 各 ~150 | 诊断脚本，各自独立 |
| `_*.py` | 各 ~80 | 快速测试脚本 |
| `figN/` | 各 4 文件 | 各实验入口和配置 |
| `tools/` | — | 工具脚本 |
| `audit/` | — | 审计脚本 |

---

## 6. 兼容性保证

### 6.1 三个兼容层文件

| 原文件 | 改为 |
|---|---|
| [common_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py) | `from common import *` |
| [train_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py) | `from train import *` |
| [eval_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py) | `from eval import *` |

### 6.2 零改动文件

以下文件的 import 语句**完全不需要修改**：

- [fig7/train_fig7.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/fig7/train_fig7.py) — `from train_experiment import TrainConfig, run_training`  ✅
- [fig7/eval_fig7.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/fig7/eval_fig7.py) — `from eval_experiment import EvalConfig, run_evaluation`  ✅
- [fig8/run_fig8.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/fig8/run_fig8.py) — `from train_experiment import ...`  ✅
- [eval_metrics.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_metrics.py) — `from common_experiment import chunk_records`  ✅
- `diag_*.py` — `from common_experiment import load_module_from_file`  ✅
- [tools/rebuild_caption_cache.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/tools/rebuild_caption_cache.py) — `from common_experiment import ...`  ✅
- 所有其他 `figN/` 入口文件  ✅

---

## 7. 内部引用更新说明

新模块内部的 import 需要使用新路径。因为 [exp/](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/experiment_bootstrap.py#88-107) 目录在 `sys.path` 上（各脚本已保证），所以子包用绝对导入：

```python
# train/protocol.py 中：
from common.model_builder import build_vlm_system     # 不是 from common_experiment
from train.phases import run_channel_phase             # 子包内引用
from train.helpers import _load_phase_best_checkpoint_strict
```

> [!IMPORTANT]
> 所有新文件（`common/`、[train/](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#971-1112)、[eval/](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#1013-1016) 内部）一律使用**新路径** import。只有三个兼容层文件使用 `from xxx import *` 做转发。

---

## 8. 实施顺序

> [!TIP]
> 建议按以下顺序实施，每步完成后验证 import 无误再继续。

### 第一步：创建 `common/` 子包

1. 创建 `common/__init__.py`
2. 创建 `common/utils.py` ← 从 [common_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py) 搬移通用工具函数
3. 创建 `common/dataset_manager.py` ← 搬移 [TaskDatasetManager](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#75-245)
4. 创建 `common/model_builder.py` ← 搬移 [build_vlm_system](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py#256-313)
5. 创建 `common/fig8_variant.py` ← 搬移 fig8 变体函数
6. 将 [common_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/common_experiment.py) 改为兼容层 `from common import *`
7. **验证**：`python -c "from common_experiment import TaskDatasetManager, build_vlm_system, chunk_records, load_module_from_file"`

### 第二步：创建 [train/](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#971-1112) 子包

1. 创建 `train/__init__.py`
2. 创建 `train/config.py` ← 搬移 [TrainConfig](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py#62-105) + 验证
3. 创建 `train/helpers.py` ← 搬移所有内部辅助函数
4. 创建 `train/phases.py` ← 搬移 3 个训练阶段
5. 创建 `train/protocol.py` ← 搬移训练协议 + train_sender
6. 创建 `train/fig8_continual.py` ← 搬移 fig8 持续学习训练
7. 创建 `train/router.py` ← 搬移入口路由
8. 将 [train_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py) 改为兼容层 `from train import *`
9. **验证**：`python -c "from train_experiment import TrainConfig, run_training, run_fig7_protocol, run_fig8_protocol"`

### 第三步：创建 [eval/](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#1013-1016) 子包

1. 创建 `eval/__init__.py`
2. 创建 `eval/config.py` ← 搬移 [EvalConfig](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py#34-81)
3. 创建 `eval/validators.py` ← 搬移校验函数
4. 创建 `eval/fig8_continual.py` ← 搬移持续学习评估
5. 创建 `eval/baselines.py` ← 搬移基线评估
6. 创建 `eval/core.py` ← 搬移通用评估核心
7. 创建 `eval/router.py` ← 搬移入口路由
8. 将 [eval_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/eval_experiment.py) 改为兼容层 `from eval import *`
9. **验证**：`python -c "from eval_experiment import EvalConfig, run_evaluation, run_fig7_eval, run_fig8_continual_evaluation"`

### 第四步：全量回归验证

```bash
cd d:\code\pycode\Semantic-Communication\VLM_CSC\exp

# 1. 兼容层 import 验证
python -c "from common_experiment import TaskDatasetManager, build_vlm_system, chunk_records, load_module_from_file, configure_runtime_logging, LABEL_MAP, collect_binary_images_from_split, collect_generic_images_from_split, resolve_fig8_variant_med_config, assert_fig8_variant_model_state"
python -c "from train_experiment import TrainConfig, run_training, run_fig7_protocol, run_fig8_protocol, run_fig9_protocol, run_fig10_protocol"
python -c "from eval_experiment import EvalConfig, run_evaluation, run_fig7_eval, run_fig8_continual_evaluation, run_fig9_eval, run_fig10_baseline_evaluation"

# 2. 新路径 import 验证
python -c "from common import TaskDatasetManager, build_vlm_system"
python -c "from train import TrainConfig, run_training"
python -c "from eval import EvalConfig, run_evaluation"

# 3. diag 脚本 import 验证
python -c "from common_experiment import load_module_from_file"

# 4. eval_metrics import 验证
python -c "from eval_metrics import evaluate_ssq_sender_snr"
```

---

## 9. 重构前后对比

| 指标 | 重构前 | 重构后 |
|---|---|---|
| 最大单文件行数 | **1800行** ([train_experiment.py](file:///d:/code/pycode/Semantic-Communication/VLM_CSC/exp/train_experiment.py)) | **~500行** (`train/phases.py`) |
| 需要理解的文件数 | 3 个大文件 | 3 个子包 × ~5 个小文件 |
| 外部 import 路径 | `from train_experiment import ...` | **不变**（兼容层） |
| 新增代码行 | — | ~90行 (`__init__.py` × 3 + 兼容层 × 3) |
| 逻辑修改 | — | **零修改**（纯搬移） |
