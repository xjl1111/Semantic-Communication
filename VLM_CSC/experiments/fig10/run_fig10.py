from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_EXP_DIR = _THIS_DIR.parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from eval import EvalConfig as _EC, run_fig10_baseline_evaluation
from common.experiment_bootstrap import apply_smoke_formal_guard, inject_provenance, make_experiment_argparser, print_experiment_params
from fig10_config import build_fig10_config
from common.paper_repro_lock import STRICT_PAPER_REPRO, validate_paper_repro_config
from train import TrainConfig as _TC, run_fig10_protocol


_REQUIRED_BASELINES = ["vlm_csc"]  # jscc/witt 为阻断项
_REQUIRED_METRICS = ["classification_accuracy", "compression_ratio", "trainable_parameters"]


def _validate_protocol(cfg: dict) -> None:
    if cfg.get("channel_type") != "awgn":
        raise RuntimeError("fig10 requires channel_type='awgn'")
    baselines = cfg.get("baselines")
    if not isinstance(baselines, list):
        raise RuntimeError("fig10 requires baselines list")
    if "vlm_csc" not in baselines:
        raise RuntimeError("fig10 baselines must include 'vlm_csc'")
    if cfg.get("metrics") != _REQUIRED_METRICS:
        raise RuntimeError(f"fig10 requires metrics={_REQUIRED_METRICS}")


def run_fig10_baseline_training(cfg: dict) -> dict:
    out_dir = Path(cfg["output_dir"]) / "vlm_csc"
    ckpt_dir = Path(cfg["checkpoint_dir"]) / "vlm_csc"
    cache_dir = Path(cfg["caption_cache_dir"]) / "vlm_csc"

    train_cfg = _TC(
        project_root=cfg["project_root"],
        model_file=cfg["model_file"],
        train_split_dir=cfg["train_split_dir"],
        output_dir=str(out_dir),
        checkpoint_dir=str(ckpt_dir),
        senders=cfg["senders"],
        blip_ckb_dir=cfg["blip_ckb_dir"],
        ram_ckb_path=cfg["ram_ckb_path"],
        sd_ckb_dir=cfg["sd_ckb_dir"],
        channel_type=cfg["channel_type"],
        seed=cfg["seed"],
        quiet_third_party=cfg["quiet_third_party"],
        strict_paper_repro=STRICT_PAPER_REPRO,
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
        caption_cache_dir=str(cache_dir),
        strict_cache_required=bool(cfg["train"].get("strict_cache_required", True)),
        train_phase_config=cfg["train"].get("train_phase_config"),
        train_tag="fig10_vlm_csc",
        fig_name="fig10",
        val_split_ratio=float(cfg.get("val_split_ratio", 0.2)),
        val_split_seed=int(cfg.get("val_split_seed", 42)),
        caption_mode=cfg.get("caption_mode", "baseline"),
        caption_prompt=cfg.get("caption_prompt"),
    )
    return run_fig10_protocol(train_cfg)


def run_fig10_baseline_eval(cfg: dict) -> dict:
    eval_cfg = _EC(
        project_root=cfg["project_root"],
        model_file=cfg["model_file"],
        target_file=cfg["target_file"],
        test_split_dir=cfg["test_split_dir"],
        output_dir=cfg["output_dir"],
        senders=cfg["senders"],
        blip_ckb_dir=cfg["blip_ckb_dir"],
        ram_ckb_path=cfg["ram_ckb_path"],
        sd_ckb_dir=cfg["sd_ckb_dir"],
        channel_type=cfg["channel_type"],
        ckpt_blip="",
        ckpt_ram="",
        snr_list=cfg["snr_list"],
        metrics=cfg["metrics"],
        sd_steps=int(cfg["sd_steps"]),
        batch_size=int(cfg["eval"]["batch_size"]),
        max_batches=int(cfg["eval"]["max_batches"]),
        max_per_class=int(cfg["eval"]["max_per_class"]),
        seed=int(cfg["seed"]),
        strict_ckpt=bool(cfg["strict_ckpt"]),
        strict_paper_repro=STRICT_PAPER_REPRO,
        quiet_third_party=bool(cfg["quiet_third_party"]),
        tag=str(cfg["eval"]["tag"]),
        required_classifier_backend="",
        baselines=cfg["baselines"],
        baseline_checkpoints=cfg["eval"].get("baseline_checkpoints"),
        export_alignment_examples=bool(cfg.get("export_alignment_examples", False)),
        fig_name="fig10",
        caption_mode=cfg.get("caption_mode", "baseline"),
        caption_prompt=cfg.get("caption_prompt"),
    )
    return run_fig10_baseline_evaluation(eval_cfg)


def _summarize_baseline_csv(csv_path: Path) -> None:
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    seen = sorted({r.get("sender", "") for r in rows})
    print(f"[FIG10_EVAL] baselines_in_csv={seen}")


def main() -> None:
    args = make_experiment_argparser("fig10").parse_args()

    cfg = build_fig10_config()
    _validate_protocol(cfg)
    validate_paper_repro_config("fig10", cfg)

    # ── smoke / formal 隔离 + 正式准入门卫 ──
    apply_smoke_formal_guard("fig10", cfg)
    inject_provenance(cfg)

    print_experiment_params("fig10", args.mode, cfg)

    if args.mode in {"train", "all"} and bool(cfg.get("train", {}).get("enabled", False)):
        reuse_best = bool(cfg.get("train", {}).get("use_previous_best_checkpoint", False))
        vlm_ckpt = Path(str(cfg.get("eval", {}).get("baseline_checkpoints", {}).get("vlm_csc", ""))).expanduser().resolve()
        if reuse_best and vlm_ckpt.exists():
            print(
                "[FIG10] train.use_previous_best_checkpoint=True and vlm_csc best checkpoint exists; "
                "skip training and reuse existing best model."
            )
        else:
            print("[FIG10_TRAIN] baseline=vlm_csc")
            train_result = run_fig10_baseline_training(cfg)
            rows = train_result.get("results", [])
            if rows:
                ckpt = rows[0].get("checkpoint") or rows[0].get("phase_joint_best_checkpoint")
                if ckpt:
                    cfg.setdefault("eval", {}).setdefault("baseline_checkpoints", {})["vlm_csc"] = str(Path(ckpt).resolve())
                    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
                    (Path(cfg["output_dir"]) / "fig10_baseline_checkpoints.json").write_text(
                        json.dumps(cfg["eval"]["baseline_checkpoints"], ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

    if args.mode in {"eval", "all"} and bool(cfg.get("eval", {}).get("enabled", False)):
        print("[FIG10_EVAL] baseline=vlm_csc")
        print("[FIG10_EVAL] baseline=jscc")
        print("[FIG10_EVAL] baseline=witt")
        eval_result = run_fig10_baseline_eval(cfg)
        curve_csv = Path(eval_result["curve_csv"])
        _summarize_baseline_csv(curve_csv)


if __name__ == "__main__":
    main()
