# VLM_CSC

本目录是当前统一版 VLM-CSC 实验框架（Fig7~Fig10）。

## 1. 环境安装

建议 Python 3.10 / CUDA 可用环境。

```bash
pip install -r VLM_CSC/requirements.txt
```

## 2. 下载模型资产（一次性）

已统一下载脚本：

```bash
python VLM_CSC/data/assets/downloaded_models/download_all_assets.py
```

默认下载到 `VLM_CSC/data/assets/downloaded_models/`：
- `blip/`
- `sd15/`
- `ram_swin_large_14m.pth`
- `recognize-anything/`

可选跳过项：
- `--skip_blip`
- `--skip_sd`
- `--skip_ram_weight`
- `--skip_ram_code`

## 3. 实验入口（当前版本）

统一核心：
- `VLM_CSC/exp/train_experiment.py`
- `VLM_CSC/exp/eval_experiment.py`
- `VLM_CSC/exp/target.py`

Figure 入口：
- Fig7: `python VLM_CSC/exp/fig7/run_fig7.py --mode all`
- Fig8: `python VLM_CSC/exp/fig8/run_fig8.py --mode all`
- Fig9: `python VLM_CSC/exp/fig9/run_fig9.py --mode all`
- Fig10: `python VLM_CSC/exp/fig10/run_fig10.py --mode all`

常用模式：
- 只训练：`--mode train`
- 只评估：`--mode eval`

## 4. 协议锁定说明

figure 入口默认启用协议锁定（防止误改关键参数）。
如仅做调试覆盖，显式加：

```bash
--allow_protocol_override
```

每次运行都会生成配置快照（含协议指纹）。

## 5. 目录建议

- `../data/datasets/`：仓库级通用数据集（供多个项目共享）
- `data/assets/downloaded_models/`：VLM_CSC 专属模型资产（统一管理）
- `data/experiments/`：VLM_CSC 专属训练产物（checkpoints/caption_cache/评估结果）
- `exp/`：仅保留实验代码与配置入口
- `model/`：仅保留模型代码与检查脚本

