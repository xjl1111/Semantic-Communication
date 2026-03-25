"""Common experiment entry helpers (deep_jscc-style layout)."""

from __future__ import annotations

import json
from typing import Callable


def run_and_report(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        print(json.dumps({"experiment": name, "status": "ok"}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"experiment": name, "status": "error", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        raise
