"""Run all available pipeline entrypoints sequentially.

Behavior:
- Always runs Step5 smoke as pipeline sanity.
- Figure scripts are delegated to `train_fig*` entrypoints so formal/proxy policy remains enforced.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.smoke_test import run_step5_smoke
from scripts.train_fig7 import main as train_fig7_main
from scripts.train_fig8 import main as train_fig8_main
from scripts.train_fig9 import main as train_fig9_main
from scripts.train_fig10 import main as train_fig10_main


def _run_with_capture(name: str, fn) -> dict:
    try:
        fn()
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    summary = {
        "mode": "proxy" if os.environ.get("VLM_CSC_ALLOW_PROXY", "0") == "1" else "formal",
    }

    summary["step5_smoke"] = {"status": "ok", "result": run_step5_smoke(seed=42)}
    summary["fig7"] = _run_with_capture("fig7", train_fig7_main)
    summary["fig8"] = _run_with_capture("fig8", train_fig8_main)
    summary["fig9"] = _run_with_capture("fig9", train_fig9_main)
    summary["fig10"] = _run_with_capture("fig10", train_fig10_main)

    print("RUN_ALL_DONE")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
