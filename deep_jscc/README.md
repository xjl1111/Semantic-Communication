# deep_jscc

本目录是 DeepJSCC 相关实验代码（AWGN / Rayleigh 等）。

## 1. 环境安装

建议 Python 3.10。

```bash
pip install -r deep_jscc/requirements.txt
```

## 2. 主要入口

实验脚本位于 `deep_jscc/experiments/`：
- `exp1_matched_snr.py`
- `exp2_snr_mismatch.py`
- `exp3_rayleigh_fading.py`

示例：

```bash
python deep_jscc/experiments/exp1_matched_snr.py
python deep_jscc/experiments/exp2_snr_mismatch.py
python deep_jscc/experiments/exp3_rayleigh_fading.py
```

## 3. 测试与快速检查

```bash
python deep_jscc/tests/test_all.py
python deep_jscc/tests/test_experiment0_sanity.py
python deep_jscc/quick_verify.py
```

## 4. 目录说明

- `../data/datasets/`：仓库级通用数据集（多项目共享）
- `data/`：deep_jscc 项目专属数据（日志/临时缓存等）
- `model/`：Encoder / Decoder / Channel
- `experiments/`：实验流程与共享工具
- `utils/`：评估指标
- `tests/`：基础回归测试

