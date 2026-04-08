from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

# Ensure VLM_CSC root is on sys.path so model imports work
_VLM_CSC_ROOT = Path(__file__).resolve().parents[2]
if str(_VLM_CSC_ROOT) not in sys.path:
    sys.path.insert(0, str(_VLM_CSC_ROOT))

STRICT_PAPER_REPRO = True
PAPER_REPRO_LOCK_VERSION = "v1"

LOCKED_MODEL_ASSUMPTIONS: Dict[str, object] = {
    "feature_dim": 128,
    "semantic_encoder_layers": 3,
    "semantic_encoder_heads": 8,
    "channel_hidden_dims": [256, 128],
    "nam_hidden_dims": [56, 128, 56, 56],
    "stm_max_size": 500,
    "med_threshold": 0.05,
    "med_tau": 10.0,
    "receiver_kb": "sd",
}


def assert_nam_hidden_dims_locked() -> None:
    """Runtime guard: verify NAM._PAPER_HIDDEN_DIMS matches the protocol lock."""
    from model.models.nam import NAM
    actual = tuple(NAM._PAPER_HIDDEN_DIMS)
    expected = tuple(LOCKED_MODEL_ASSUMPTIONS["nam_hidden_dims"])
    if actual != expected:
        raise RuntimeError(
            f"NAM hidden dims protocol violation: code has {actual}, lock requires {expected}. "
            f"Someone modified NAM._PAPER_HIDDEN_DIMS."
        )

LOCKED_FIG_PROTOCOLS: Dict[str, Dict[str, object]] = {
    "fig7": {
        "name": "fig7_awgn_catsvsdogs_ssq_v1",
        "required_senders": ["blip", "ram"],
        "required_metrics": ["ssq"],
        "required_classifier_backend": ["clip_zeroshot", "clip_finetuned"],
        "require_strict_ckpt": True,
        "channel_type": "awgn",
    },
    "fig8": {
        "name": "fig8_rayleigh_continual_bleu_v1",
        "required_metrics": ["bleu1", "bleu2"],
        "require_strict_ckpt": True,
        "channel_type": "rayleigh",
        "required_dataset_sequence": ["cifar", "birds", "catsvsdogs"],
        "require_med_toggle": True,
        "required_eval_output_mode": "continual_learning_map",
    },
    "fig9": {
        "name": "fig9_awgn_nam_bleu_v1",
        "required_metrics": ["bleu1", "bleu2"],
        "require_strict_ckpt": True,
        "channel_type": "awgn",
        "with_nam_train_mode": "uniform_range",
        "with_nam_train_range": [0.0, 10.0],
        "without_nam_train_points": [0.0, 2.0, 4.0, 8.0],
    },
    "fig10": {
        "name": "fig10_awgn_main_performance_v1",
        "required_metrics": ["classification_accuracy", "compression_ratio", "trainable_parameters"],
        "require_strict_ckpt": True,
        "channel_type": "awgn",
        "required_dataset": "catsvsdogs",
        "required_baselines": ["vlm_csc"],  # jscc/witt 为阻断项，见 BLOCKING_ITEMS.md
    },
}


def validate_paper_repro_config(fig_name: str, cfg: dict) -> None:
    if not STRICT_PAPER_REPRO:
        return

    # Runtime check: NAM hidden dims must match paper specification
    assert_nam_hidden_dims_locked()

    if fig_name not in LOCKED_FIG_PROTOCOLS:
        raise RuntimeError(f"No paper repro lock found for figure: {fig_name}")

    lock = LOCKED_FIG_PROTOCOLS[fig_name]
    protocol = cfg.get("protocol", {})

    if protocol.get("name") != lock["name"]:
        raise RuntimeError(
            f"{fig_name} paper lock violation: protocol.name must be {lock['name']}, got {protocol.get('name')}"
        )

    required_senders = lock.get("required_senders")
    cfg_senders = cfg.get("senders", [])
    if required_senders is not None:
        # 允许 cfg senders 是 required_senders 的子集（如仅 BLIP）
        if not set(cfg_senders).issubset(set(required_senders)):
            raise RuntimeError(f"{fig_name} paper lock violation: senders must be subset of {required_senders}, got {cfg_senders}")

    if cfg.get("metrics") != lock["required_metrics"]:
        raise RuntimeError(
            f"{fig_name} paper lock violation: metrics must be {lock['required_metrics']}, got {cfg.get('metrics')}"
        )

    if cfg.get("channel_type") != lock["channel_type"]:
        raise RuntimeError(
            f"{fig_name} paper lock violation: channel_type must be {lock['channel_type']}, got {cfg.get('channel_type')}"
        )

    if bool(cfg.get("strict_ckpt")) != bool(lock["require_strict_ckpt"]):
        raise RuntimeError(f"{fig_name} paper lock violation: strict_ckpt must be enabled.")

    required_backend = lock.get("required_classifier_backend")
    if required_backend is not None:
        actual_backend = protocol.get("required_classifier_backend")
        allowed = required_backend if isinstance(required_backend, list) else [required_backend]
        if actual_backend not in allowed:
            raise RuntimeError(
                f"{fig_name} paper lock violation: required_classifier_backend must be one of {allowed}, "
                f"got {actual_backend}"
            )

    if protocol.get("receiver_kb") != LOCKED_MODEL_ASSUMPTIONS["receiver_kb"]:
        raise RuntimeError(
            f"{fig_name} paper lock violation: receiver_kb must be {LOCKED_MODEL_ASSUMPTIONS['receiver_kb']}, "
            f"got {protocol.get('receiver_kb')}"
        )

    if "required_dataset_sequence" in lock:
        if cfg.get("dataset_sequence") != lock["required_dataset_sequence"]:
            raise RuntimeError(
                f"{fig_name} paper lock violation: dataset_sequence must be {lock['required_dataset_sequence']}, "
                f"got {cfg.get('dataset_sequence')}"
            )

    if "required_dataset" in lock:
        if cfg.get("dataset") != lock["required_dataset"]:
            raise RuntimeError(
                f"{fig_name} paper lock violation: dataset must be {lock['required_dataset']}, got {cfg.get('dataset')}"
            )

    if "required_baselines" in lock:
        if cfg.get("baselines") != lock["required_baselines"]:
            raise RuntimeError(
                f"{fig_name} paper lock violation: baselines must be {lock['required_baselines']}, "
                f"got {cfg.get('baselines')}"
            )

    if lock.get("require_med_toggle", False):
        med_flags = cfg.get("med_variants")
        if med_flags == [True, False]:
            pass
        elif med_flags == ["with_med", "without_med"]:
            pass
        else:
            raise RuntimeError(
                f"{fig_name} paper lock violation: med_variants must be [True, False] or ['with_med','without_med'], got {med_flags}"
            )

    if "required_eval_output_mode" in lock:
        if cfg.get("eval_output_mode") != lock["required_eval_output_mode"]:
            raise RuntimeError(
                f"{fig_name} paper lock violation: eval_output_mode must be {lock['required_eval_output_mode']}, "
                f"got {cfg.get('eval_output_mode')}"
            )

    if fig_name == "fig9":
        nam_experiments_raw = cfg.get("nam_experiments", [])
        if isinstance(nam_experiments_raw, dict):
            nam_experiments = []
            for exp_name, exp_cfg in nam_experiments_raw.items():
                if not isinstance(exp_cfg, dict):
                    raise RuntimeError(f"fig9 paper lock violation: nam_experiments[{exp_name}] must be a dict")
                item = dict(exp_cfg)
                item.setdefault("name", exp_name)
                nam_experiments.append(item)
        elif isinstance(nam_experiments_raw, list):
            nam_experiments = nam_experiments_raw
        else:
            raise RuntimeError("fig9 paper lock violation: nam_experiments must be a non-empty list or dict")

        if len(nam_experiments) == 0:
            raise RuntimeError("fig9 paper lock violation: nam_experiments must be non-empty")

        with_nam = [x for x in nam_experiments if bool(x.get("use_nam", False))]
        without_nam = [x for x in nam_experiments if not bool(x.get("use_nam", False))]
        if len(with_nam) != 1:
            raise RuntimeError("fig9 paper lock violation: exactly one with-NAM experiment is required")

        with_nam_cfg = with_nam[0]
        if with_nam_cfg.get("train_snr_mode") != lock["with_nam_train_mode"]:
            raise RuntimeError(
                f"fig9 paper lock violation: with-NAM train_snr_mode must be {lock['with_nam_train_mode']}, "
                f"got {with_nam_cfg.get('train_snr_mode')}"
            )
        with_range = [float(with_nam_cfg.get("train_snr_min_db", -999)), float(with_nam_cfg.get("train_snr_max_db", -999))]
        if with_range != [float(lock["with_nam_train_range"][0]), float(lock["with_nam_train_range"][1])]:
            raise RuntimeError(
                f"fig9 paper lock violation: with-NAM train SNR range must be {lock['with_nam_train_range']}, "
                f"got {with_range}"
            )

        without_points = sorted(float(x.get("train_snr_db", -999)) for x in without_nam)
        if without_points != [float(v) for v in lock["without_nam_train_points"]]:
            raise RuntimeError(
                f"fig9 paper lock violation: without-NAM train points must be {lock['without_nam_train_points']}, "
                f"got {without_points}"
            )
