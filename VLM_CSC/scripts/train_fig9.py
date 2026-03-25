"""Train pipeline entry for Fig.9."""

from __future__ import annotations

import json
import os

from eval.eval_fig9 import run_fig9_eval


def main() -> None:
    if os.environ.get("VLM_CSC_ALLOW_PROXY", "0") != "1":
        raise RuntimeError(
            "Proxy mode is disabled for formal Fig.9 training. "
            "Set VLM_CSC_ALLOW_PROXY=1 only for pipeline checks."
        )
    result = run_fig9_eval(
        snr_test_db=tuple(range(0, 11)),
        nam_off_train_snrs=(0, 2, 4, 8),
        output_root="outputs/fig9",
    )
    print("FIG9_RUN_OK")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
