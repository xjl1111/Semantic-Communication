"""
Fig.9 — 仅训练入口

用法：
    python train_fig9.py
    python train_fig9.py --train_max_per_class 100   # 快速验证
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_EXP_DIR = _THIS_DIR.parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from run_fig9 import main as _run_fig9_main

# 强制 mode=train
sys.argv = [a for a in sys.argv if a not in ("--mode", "train", "eval", "all")]
sys.argv.insert(1, "--mode")
sys.argv.insert(2, "train")

if __name__ == "__main__":
    _run_fig9_main()
