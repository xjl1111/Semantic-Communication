"""VLM_CSC 训练子包。"""

from train.config import TrainConfig
from train.helpers import (
    _compute_bleu_n,
    _enrich_saved_checkpoints,
    _evaluate_bleu_on_records,
    _load_phase_best_checkpoint_strict,
    _prepare_merged_batch,
    _resolve_train_snr,
    _run_joint_train_step_merged,
    _run_semantic_train_step_merged,
    _sample_memory_batch_for_step,
    _snr_sampler,
    _train_text_matches_label,
    _update_med_and_check,
    _validate_fig8_variant_checkpoint_map_complete,
    _write_matrix_csv,
)
from train.phases import run_channel_phase, run_joint_phase, run_semantic_phase
from train.protocol import run_paper_training_protocol, train_sender
from train.fig8_continual import run_fig8_continual_training
from train.config import set_seed, load_vlm_module, _validate_train_phase_config
from train.router import (
    run_training,
    run_fig7_protocol,
    run_fig8_protocol,
    run_fig9_protocol,
    run_fig10_protocol,
    _run_training_core,
)

__all__ = [
    "TrainConfig",
    "set_seed",
    "load_vlm_module",
    "_validate_train_phase_config",
    "_train_text_matches_label",
    "_snr_sampler",
    "_resolve_train_snr",
    "_load_phase_best_checkpoint_strict",
    "_sample_memory_batch_for_step",
    "_prepare_merged_batch",
    "_update_med_and_check",
    "_run_semantic_train_step_merged",
    "_run_joint_train_step_merged",
    "_enrich_saved_checkpoints",
    "_compute_bleu_n",
    "_evaluate_bleu_on_records",
    "_write_matrix_csv",
    "_validate_fig8_variant_checkpoint_map_complete",
    "run_channel_phase",
    "run_semantic_phase",
    "run_joint_phase",
    "run_paper_training_protocol",
    "train_sender",
    "run_fig8_continual_training",
    "_run_training_core",
    "run_fig7_protocol",
    "run_fig8_protocol",
    "run_fig9_protocol",
    "run_fig10_protocol",
    "run_training",
]
