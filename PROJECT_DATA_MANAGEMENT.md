# 项目数据分级管理规范（仓库级）

本文件用于统一本仓库所有子项目的数据目录结构，确保“公共数据可复用、项目数据不混放”。

## 1. 目标

- 公共数据集统一管理，避免重复下载与重复拷贝。
- 子项目专属数据独立管理，避免相互污染。
- 新增项目可直接按本规范落地，无需重复设计目录。

## 2. 统一目录分层

仓库根目录统一采用两层数据体系：

- `data/datasets/`：仓库级通用数据集（可被多个项目共享）。
- `<project>/data/`：项目级专属数据（仅该项目使用）。

## 3. 各层允许存放内容

### 3.1 仓库级通用数据（`data/datasets/`）

允许：

- 通用公开数据集原始文件
- 通用标准划分数据（train/val/test）

禁止：

- 任一项目专属的缓存、日志、checkpoint、中间产物

### 3.2 项目级专属数据（`<project>/data/`）

建议标准子目录：

- `assets/`：该项目专属模型资产（权重、外部依赖文件）
- `experiments/`：该项目训练与评估产物（checkpoints、csv、可视化等）
- `cache/`（可选）：该项目临时缓存
- `logs/`（可选）：该项目日志

禁止：

- 跨项目共享的通用数据集

## 4. 新项目接入模板

新增项目 `<new_project>` 时，按以下步骤执行：

1. 创建项目私有数据目录：
   - `<new_project>/data/`
   - `<new_project>/data/assets/`
   - `<new_project>/data/experiments/`
2. 在 `<new_project>/data/README.md` 写明“仅项目私有数据”。
3. 代码中将通用数据集路径指向：`<repo>/data/datasets/`。
4. 代码中将项目产物路径指向：`<repo>/<new_project>/data/` 下。
5. 禁止在 `<new_project>/src|model|exp|tests` 中写入真实数据文件。

## 5. 路径约定建议（Python）

建议统一使用 `Path` 构造，避免硬编码绝对路径：

- `REPO_ROOT = Path(__file__).resolve().parents[N]`
- `COMMON_DATASETS_DIR = REPO_ROOT / "data" / "datasets"`
- `PROJECT_DATA_DIR = REPO_ROOT / "<project>" / "data"`

## 6. 变更检查清单（PR自查）

提交前至少检查：

- [ ] 新增数据集是否放在 `data/datasets/`
- [ ] 新增 checkpoint/缓存/日志 是否放在 `<project>/data/`
- [ ] 是否存在硬编码绝对路径
- [ ] `exp/`、`model/`、`tests/` 是否只含代码
- [ ] 项目 README 是否描述了数据目录规则

## 7. 适用范围

本规范适用于本仓库所有子项目（当前包括 `VLM_CSC`、`deep_jscc`，以及后续新增项目）。
