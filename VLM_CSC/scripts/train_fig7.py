"""Train pipeline entry for Fig.7."""

from __future__ import annotations

import json
import os

from eval.eval_fig7 import run_fig7_eval


def main() -> None:
    if os.environ.get("VLM_CSC_ALLOW_PROXY", "0") != "1":
        raise RuntimeError(
            "Proxy mode is disabled for formal Fig.7 training. "
            "Set VLM_CSC_ALLOW_PROXY=1 only for pipeline checks."
        )
    result = run_fig7_eval(
        sender_models=("blip", "lemon", "ram"),
        snr_test_db=tuple(range(0, 11)),
        output_root="outputs/fig7",
        dataset_name="catsvsdogs",
        channel_name="awgn",
    )
    print("FIG7_RUN_OK")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
