"""VLM_CSC 评估子包。"""

from eval.config import EvalConfig, load_module
from eval.validators import (
    _file_sha256,
    _get_fig8_variant_checkpoint,
    _normalize_fig8_med_variants,
    _validate_fig8_strict_protocol_inputs,
    check_sd_assets,
)
from eval.fig8_continual import _run_continual_bleu_map
from eval.baselines import _build_jscc_pipeline, _run_baseline_performance
from eval.core import _run_evaluation_core
from eval.router import (
    run_evaluation,
    run_fig7_eval,
    run_fig8_continual_evaluation,
    run_fig9_eval,
    run_fig10_baseline_evaluation,
)

__all__ = [
    "EvalConfig",
    "load_module",
    "_file_sha256",
    "check_sd_assets",
    "_get_fig8_variant_checkpoint",
    "_normalize_fig8_med_variants",
    "_validate_fig8_strict_protocol_inputs",
    "_run_continual_bleu_map",
    "_build_jscc_pipeline",
    "_run_baseline_performance",
    "_run_evaluation_core",
    "run_evaluation",
    "run_fig7_eval",
    "run_fig8_continual_evaluation",
    "run_fig9_eval",
    "run_fig10_baseline_evaluation",
]
