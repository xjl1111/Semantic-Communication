"""Train pipeline entry for Fig.8."""

from __future__ import annotations

import json
import os

from eval.eval_fig8 import run_fig8_eval


def main() -> None:
    if os.environ.get("VLM_CSC_ALLOW_PROXY", "0") != "1":
        raise RuntimeError(
            "Proxy mode is disabled for formal Fig.8 training. "
            "Set VLM_CSC_ALLOW_PROXY=1 only for pipeline checks."
        )
    result = run_fig8_eval(
        task_order=("cifar", "birds", "catsvsdogs"),
        output_root="outputs/fig8",
        channel_name="rayleigh",
    )
    print("FIG8_RUN_OK")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
