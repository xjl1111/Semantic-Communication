"""Train pipeline entry for Fig.10."""

from __future__ import annotations

import json
import os

from eval.eval_fig10 import run_fig10_eval


def main() -> None:
    if os.environ.get("VLM_CSC_ALLOW_PROXY", "0") != "1":
        raise RuntimeError(
            "Proxy mode is disabled for formal Fig.10 training. "
            "Set VLM_CSC_ALLOW_PROXY=1 only for pipeline checks."
        )
    result = run_fig10_eval(output_root="outputs/fig10", image_size=224)
    print("FIG10_RUN_OK")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
