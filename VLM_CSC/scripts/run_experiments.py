"""Unified experiment runner.

Features:
- Run all experiments in one command
- Run a single figure via CLI
- Control formal/proxy mode from one place
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_fig7 import main as run_fig7_main
from scripts.run_fig8 import main as run_fig8_main
from scripts.run_fig9 import main as run_fig9_main
from scripts.run_fig10 import main as run_fig10_main
from scripts.train_all import main as train_all_main


def _execute(name: str, fn: Callable[[], None]) -> dict:
    try:
        fn()
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all figure experiments")
    parser.add_argument("--fig", choices=["fig7", "fig8", "fig9", "fig10", "train"], help="Run one target")
    parser.add_argument("--mode", choices=["formal", "proxy"], default="formal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "proxy":
        os.environ["VLM_CSC_ALLOW_PROXY"] = "1"
    else:
        os.environ["VLM_CSC_ALLOW_PROXY"] = "0"

    dispatch: dict[str, Callable[[], None]] = {
        "train": train_all_main,
        "fig7": run_fig7_main,
        "fig8": run_fig8_main,
        "fig9": run_fig9_main,
        "fig10": run_fig10_main,
    }

    if not args.all and not args.fig:
        raise ValueError("Specify --all or --fig")

    summary: dict[str, object] = {"mode": args.mode}
    if args.all:
        for key in ("fig7", "fig8", "fig9", "fig10"):
            summary[key] = _execute(key, dispatch[key])
    else:
        summary[args.fig] = _execute(args.fig, dispatch[args.fig])

    print("RUN_EXPERIMENTS_DONE")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
