"""
Fig.7 — 仅评估入口

用法：
    python eval_fig7.py                                # 默认配置
    python eval_fig7.py --max_per_class 50             # 快速验证（5× 加速）
    python eval_fig7.py --sd_steps 10                  # 减少 DDIM 步数（3× 加速）
    python eval_fig7.py --sd_steps 10 --max_per_class 50  # 叠加（≈15× 加速）

注意：评估需要训练好的 checkpoint；若 checkpoint 不存在会自动触发训练。
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_EXP_DIR = _THIS_DIR.parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from run_fig7 import main as _run_fig7_main

# 强制 mode=eval
sys.argv = [a for a in sys.argv if a not in ("--mode", "train", "eval", "all")]
sys.argv.insert(1, "--mode")
sys.argv.insert(2, "eval")

if __name__ == "__main__":
    _run_fig7_main()
