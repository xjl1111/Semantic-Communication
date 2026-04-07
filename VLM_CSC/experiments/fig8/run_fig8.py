from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_EXP_DIR = _THIS_DIR.parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from eval import EvalConfig as _EC, run_fig8_continual_evaluation
from common.experiment_bootstrap import apply_smoke_formal_guard, inject_provenance, make_experiment_argparser, print_experiment_params
from fig8_config import build_fig8_config
from common.paper_repro_lock import STRICT_PAPER_REPRO, validate_paper_repro_config
from train import TrainConfig as _TC, run_fig8_continual_training

_REQUIRED_VARIANTS = ("with_med", "without_med")
_MAP_FILE_NAME = "fig8_variant_checkpoint_map.json"


def _normalize_med_variants(raw_variants) -> list[bool]:
    if not isinstance(raw_variants, list):
        raise RuntimeError("fig8 med_variants must be a list")
    norm = []
    for item in raw_variants:
        if isinstance(item, bool):
            norm.append(item)
            continue
        name = str(item).strip().lower()
        if name == "with_med":
            norm.append(True)
        elif name == "without_med":
            norm.append(False)
        else:
            raise RuntimeError(f"Unsupported fig8 med variant: {item}")
    if norm != [True, False]:
        raise RuntimeError("fig8 requires med_variants=['with_med','without_med']")
    return norm


def _normalize_and_validate_checkpoint_map(cfg: dict) -> dict:
    dataset_sequence: list[str] | None = cfg.get("dataset_sequence")
    if dataset_sequence != ["cifar", "birds", "catsvsdogs"]:
        raise RuntimeError("fig8 requires dataset_sequence=['cifar','birds','catsvsdogs']")
    # 此处 dataset_sequence 已确认为 ["cifar","birds","catsvsdogs"]
    assert isinstance(dataset_sequence, list)

    eval_cfg = cfg.get("eval", {})
    raw_map = eval_cfg.get("fig8_variant_checkpoint_map")
    if not isinstance(raw_map, dict):
        raise RuntimeError("fig8 requires eval.fig8_variant_checkpoint_map as a dict")

    senders = [str(s).strip().lower() for s in cfg.get("senders", [])]
    if not senders:
        raise RuntimeError("fig8 requires non-empty senders")

    normalized: dict = {}
    for variant in _REQUIRED_VARIANTS:
        variant_block = raw_map.get(variant)
        if not isinstance(variant_block, dict):
            raise RuntimeError(f"fig8 map missing variant={variant}")
        normalized[variant] = {}

        for sender in senders:
            sender_block = variant_block.get(sender)
            if not isinstance(sender_block, dict):
                raise RuntimeError(f"fig8 map missing sender={sender} in variant={variant}")

            missing_tasks = [task for task in dataset_sequence if task not in sender_block]
            if missing_tasks:
                raise RuntimeError(
                    f"fig8 map missing tasks variant={variant}, sender={sender}, tasks={missing_tasks}"
                )

            normalized[variant][sender] = {}
            for task in dataset_sequence:
                ckpt = Path(str(sender_block[task])).expanduser().resolve()
                if not ckpt.exists():
                    raise RuntimeError(
                        f"fig8 checkpoint missing variant={variant}, sender={sender}, task={task}: {ckpt}"
                    )
                normalized[variant][sender][task] = str(ckpt)

    return normalized


def _resolve_checkpoint_map(cfg: dict) -> dict:
    eval_cfg = cfg.setdefault("eval", {})
    if isinstance(eval_cfg.get("fig8_variant_checkpoint_map"), dict):
        return _normalize_and_validate_checkpoint_map(cfg)

    map_path = Path(cfg["train_monitor_output_dir"]) / _MAP_FILE_NAME
    if not map_path.exists():
        raise RuntimeError(
            "fig8 evaluation requires checkpoint map from either eval.fig8_variant_checkpoint_map or output json"
        )
    loaded = json.loads(map_path.read_text(encoding="utf-8"))
    eval_cfg["fig8_variant_checkpoint_map"] = loaded
    return _normalize_and_validate_checkpoint_map(cfg)


def _validate_protocol(cfg: dict, mode: str) -> None:
    if "train_monitor_output_dir" not in cfg or "final_eval_output_dir" not in cfg:
        raise RuntimeError("fig8 requires train_monitor_output_dir and final_eval_output_dir in config")
    if str(cfg["train_monitor_output_dir"]).strip() == "" or str(cfg["final_eval_output_dir"]).strip() == "":
        raise RuntimeError("fig8 output directories cannot be empty")
    if Path(cfg["train_monitor_output_dir"]).resolve() == Path(cfg["final_eval_output_dir"]).resolve():
        raise RuntimeError("fig8 train_monitor_output_dir and final_eval_output_dir must be different")

    if cfg.get("channel_type") != "rayleigh":
        raise RuntimeError("fig8 requires channel_type='rayleigh'")
    if cfg.get("metrics") != ["bleu1", "bleu2"]:
        raise RuntimeError("fig8 requires metrics=['bleu1','bleu2']")
    if cfg.get("eval_output_mode") != "continual_learning_map":
        raise RuntimeError("fig8 requires eval_output_mode='continual_learning_map'")
    if not isinstance(cfg.get("dataset_roots"), dict) or not isinstance(cfg.get("dataset_splits"), dict):
        raise RuntimeError("fig8 requires dataset_roots and dataset_splits")
    if cfg.get("dataset_sequence") != ["cifar", "birds", "catsvsdogs"]:
        raise RuntimeError("fig8 requires dataset_sequence=['cifar','birds','catsvsdogs']")
    _normalize_med_variants(cfg.get("med_variants"))

    if mode in {"eval", "all"}:
        snr = cfg.get("fig8_eval_snr_db", cfg.get("eval", {}).get("fig8_eval_snr_db"))
        if snr is None:
            raise RuntimeError("fig8 requires fig8_eval_snr_db")
        if isinstance(snr, (dict, list, tuple)):
            raise RuntimeError("fig8_eval_snr_db must be a single numeric value")


def _snapshot_config(cfg: dict, mode: str) -> Path:
    if mode in {"train"}:
        output_dir = Path(cfg["train_monitor_output_dir"])
    elif mode in {"eval"}:
        output_dir = Path(cfg["final_eval_output_dir"])
    else:
        output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = cfg.get("eval", {}).get("tag", "fig8_eval") if mode in {"eval", "all"} else cfg.get("train", {}).get("train_tag", "fig8_train")

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "protocol_name": cfg.get("protocol", {}).get("name", "fig8"),
        "config": cfg,
    }
    payload["protocol_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "senders": cfg.get("senders"),
                "metrics": cfg.get("metrics"),
                "channel_type": cfg.get("channel_type"),
                "dataset_sequence": cfg.get("dataset_sequence"),
                "eval_output_mode": cfg.get("eval_output_mode"),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    out = output_dir / f"fig8_{run_tag}_config_snapshot.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 协议 YAML 输出 ──
    try:
        import yaml
        protocol_yaml = {
            "protocol_name": cfg.get("protocol", {}).get("name", "fig8"),
            "channel_type": cfg.get("channel_type"),
            "metrics": cfg.get("metrics"),
            "dataset_sequence": cfg.get("dataset_sequence"),
            "med_variants": cfg.get("med_variants"),
            "eval_output_mode": cfg.get("eval_output_mode"),
            "senders": cfg.get("senders"),
            "med_params": {
                "stm_max_size": cfg.get("train", {}).get("med_kwargs", {}).get("stm_max_size"),
                "tau": cfg.get("train", {}).get("med_kwargs", {}).get("tau"),
                "threshold": cfg.get("train", {}).get("med_kwargs", {}).get("threshold"),
                "transfer_if": cfg.get("train", {}).get("med_kwargs", {}).get("transfer_if"),
            },
            "training": {
                "snr_mode": cfg.get("train", {}).get("train_snr_mode"),
                "snr_db": cfg.get("train", {}).get("train_snr_min_db"),
                "lr": cfg.get("train", {}).get("train_lr"),
                "batch_size": cfg.get("train", {}).get("train_batch_size"),
                "channel_epochs": cfg.get("train", {}).get("train_phase_config", {}).get("channel_phase", {}).get("epochs"),
                "semantic_epochs": cfg.get("train", {}).get("train_phase_config", {}).get("semantic_phase", {}).get("epochs"),
                "joint_max_epochs": cfg.get("train", {}).get("train_phase_config", {}).get("joint_phase", {}).get("max_joint_epochs"),
                "joint_alpha": cfg.get("train", {}).get("train_phase_config", {}).get("joint_phase", {}).get("alpha"),
                "joint_beta": cfg.get("train", {}).get("train_phase_config", {}).get("joint_phase", {}).get("beta"),
            },
            "evaluation": {
                "snr_db": cfg.get("fig8_eval_snr_db"),
                "sd_steps": cfg.get("sd_steps"),
            },
            "fingerprint_sha256": payload.get("protocol_fingerprint_sha256"),
        }
        yaml_out = output_dir / "fig8_protocol.yaml"
        yaml_out.write_text(
            yaml.dump(protocol_yaml, allow_unicode=True, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
    except ImportError:
        pass  # PyYAML not installed, skip YAML output

    return out


def _build_train_config(cfg: dict) -> _TC:
    return _TC(
        project_root=cfg["project_root"],
        model_file=cfg["model_file"],
        train_split_dir="",
        output_dir=cfg["train_monitor_output_dir"],
        checkpoint_dir=cfg["checkpoint_dir"],
        senders=cfg["senders"],
        blip_ckb_dir=cfg["blip_ckb_dir"],
        ram_ckb_path=cfg["ram_ckb_path"],
        sd_ckb_dir=cfg["sd_ckb_dir"],
        channel_type=cfg["channel_type"],
        seed=cfg["seed"],
        quiet_third_party=cfg["quiet_third_party"],
        strict_paper_repro=STRICT_PAPER_REPRO,
        max_text_len=int(cfg.get("max_text_len", 24)),
        max_text_len_by_sender=cfg.get("max_text_len_by_sender"),
        train_epochs=int(cfg.get("train", {}).get("train_epochs", 1)),
        train_lr=float(cfg["train"]["train_lr"]),
        train_weight_decay=float(cfg["train"]["train_weight_decay"]),
        train_batch_size=int(cfg["train"]["train_batch_size"]),
        train_max_batches=int(cfg["train"]["train_max_batches"]),
        train_max_per_class=int(cfg["train"]["train_max_per_class"]),
        val_ratio=float(cfg["train"]["val_ratio"]),
        train_snr_min_db=float(cfg["train"]["train_snr_min_db"]),
        train_snr_max_db=float(cfg["train"]["train_snr_max_db"]),
        train_snr_mode=str(cfg["train"]["train_snr_mode"]),
        use_caption_cache=bool(cfg["train"]["use_caption_cache"]),
        caption_cache_dir=cfg["caption_cache_dir"],
        strict_cache_required=bool(cfg["train"].get("strict_cache_required", True)),
        train_phase_config=cfg["train"].get("train_phase_config"),
        train_tag=str(cfg["train"]["train_tag"]),
        fig_name="fig8",
        dataset_sequence=cfg.get("dataset_sequence"),
        dataset_roots=cfg.get("dataset_roots"),
        dataset_splits=cfg.get("dataset_splits"),
        val_split_ratio=float(cfg.get("val_split_ratio", 0.2)),
        val_split_seed=int(cfg.get("val_split_seed", 42)),
        med_kwargs=cfg.get("train", {}).get("med_kwargs"),
        caption_mode=cfg.get("caption_mode", "baseline"),
        caption_prompt=cfg.get("caption_prompt"),
        channel_dim=cfg.get("channel_dim"),
        sd_steps=int(cfg.get("sd_steps", 20)),
    )


def _build_eval_config(cfg: dict) -> _EC:
    eval_snr = cfg.get("fig8_eval_snr_db", cfg.get("eval", {}).get("fig8_eval_snr_db"))
    return _EC(
        project_root=cfg["project_root"],
        model_file=cfg["model_file"],
        target_file=cfg["target_file"],
        test_split_dir="",
        output_dir=cfg["final_eval_output_dir"],
        senders=cfg["senders"],
        blip_ckb_dir=cfg["blip_ckb_dir"],
        ram_ckb_path=cfg["ram_ckb_path"],
        sd_ckb_dir=cfg["sd_ckb_dir"],
        channel_type=cfg["channel_type"],
        ckpt_blip="",
        ckpt_ram="",
        snr_list=cfg.get("snr_list", [float(eval_snr)]),
        metrics=cfg.get("metrics", ["bleu1", "bleu2"]),
        sd_steps=int(cfg["sd_steps"]),
        sd_height=int(cfg.get("sd_height", 512)),
        sd_width=int(cfg.get("sd_width", 512)),
        batch_size=int(cfg["eval"]["batch_size"]),
        max_batches=int(cfg["eval"]["max_batches"]),
        max_per_class=int(cfg["eval"]["max_per_class"]),
        seed=int(cfg["seed"]),
        strict_ckpt=bool(cfg["strict_ckpt"]),
        strict_paper_repro=STRICT_PAPER_REPRO,
        quiet_third_party=bool(cfg["quiet_third_party"]),
        tag=str(cfg["eval"]["tag"]),
        required_classifier_backend="clip_zeroshot",
        max_text_len=int(cfg.get("max_text_len", 24)),
        max_text_len_by_sender=cfg.get("max_text_len_by_sender"),
        channel_dim=cfg.get("channel_dim"),
        dataset_sequence=cfg.get("dataset_sequence"),
        med_variants=cfg.get("med_variants"),
        eval_output_mode=cfg.get("eval_output_mode"),
        fig_name="fig8",
        fig8_variant_checkpoint_map=cfg.get("eval", {}).get("fig8_variant_checkpoint_map"),
        dataset_roots=cfg.get("dataset_roots"),
        dataset_splits=cfg.get("dataset_splits"),
        fig8_eval_snr_db=float(eval_snr),
        training_snr_protocol=(
            f"mode={cfg.get('train', {}).get('train_snr_mode')} "
            f"min={cfg.get('train', {}).get('train_snr_min_db')} "
            f"max={cfg.get('train', {}).get('train_snr_max_db')}"
        ),
        med_kwargs=cfg.get("train", {}).get("med_kwargs"),
        caption_mode=cfg.get("caption_mode", "baseline"),
        caption_prompt=cfg.get("caption_prompt"),
    )


def _has_fig8_best_checkpoint_map(cfg: dict) -> bool:
    raw_map = cfg.get("eval", {}).get("fig8_variant_checkpoint_map")
    if not isinstance(raw_map, dict):
        return False
    for variant in ["with_med", "without_med"]:
        variant_block = raw_map.get(variant)
        if not isinstance(variant_block, dict):
            return False
        for sender in cfg.get("senders", []):
            sender_block = variant_block.get(sender)
            if not isinstance(sender_block, dict):
                return False
            for task in ["cifar", "birds", "catsvsdogs"]:
                ckpt = Path(str(sender_block.get(task, ""))).expanduser().resolve()
                if not ckpt.exists():
                    return False
    return True


def main() -> None:
    args = make_experiment_argparser(
        "fig8",
        extra_args=[
            ("--allow_protocol_override", {"action": "store_true",
                                           "help": "允许覆盖协议锁定参数（仅限调试）"}),
            ("--sd_steps",       {"type": int,   "default": None,
                                  "help": "DDIM 去噪步数（覆盖配置）"}),
            ("--max_per_class",  {"type": int,   "default": None,
                                  "help": "评估时每类最大样本数（-1=全部）"}),
            ("--batch_size",     {"type": int,   "default": None,
                                  "help": "评估/训练 batch 大小"}),
            ("--train_max_per_class", {"type": int, "default": None,
                                       "help": "训练时每类最大样本数（-1=全部）"}),
        ],
    ).parse_args()

    cfg = build_fig8_config()

    # ── CLI 覆盖 ──
    if args.sd_steps is not None:
        cfg["sd_steps"] = args.sd_steps
    if args.max_per_class is not None:
        cfg["eval"]["max_per_class"] = args.max_per_class
    if args.batch_size is not None:
        cfg["eval"]["batch_size"] = args.batch_size
        cfg["train"]["train_batch_size"] = args.batch_size
    if args.train_max_per_class is not None:
        cfg["train"]["train_max_per_class"] = args.train_max_per_class

    _validate_protocol(cfg, args.mode)
    validate_paper_repro_config("fig8", cfg)

    # ── smoke / formal 隔离 + 正式准入门卫 ──
    apply_smoke_formal_guard(
        "fig8", cfg,
        output_dirs=["train_monitor_output_dir", "final_eval_output_dir"],
    )
    inject_provenance(cfg)

    print_experiment_params("fig8", args.mode, cfg)

    snapshot = _snapshot_config(cfg, args.mode)

    if args.mode in {"train", "all"} and bool(cfg.get("train", {}).get("enabled", False)):
        reuse_best = bool(cfg.get("train", {}).get("use_previous_best_checkpoint", False))
        if reuse_best and _has_fig8_best_checkpoint_map(cfg):
            pass
        else:
            train_cfg = _build_train_config(cfg)
            run_fig8_continual_training(config=train_cfg, device="cuda")

    if args.mode in {"eval", "all"} and bool(cfg.get("eval", {}).get("enabled", False)):
        cfg.setdefault("eval", {})["fig8_variant_checkpoint_map"] = _resolve_checkpoint_map(cfg)
        eval_cfg = _build_eval_config(cfg)
        run_fig8_continual_evaluation(eval_cfg)


if __name__ == "__main__":
    main()
