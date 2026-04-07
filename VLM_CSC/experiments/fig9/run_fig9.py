from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_EXP_DIR = _THIS_DIR.parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from eval import EvalConfig as _EC, run_fig9_eval
from common.experiment_bootstrap import apply_smoke_formal_guard, inject_provenance, make_experiment_argparser, print_experiment_params
from fig9_config import build_fig9_config
from common.paper_repro_lock import STRICT_PAPER_REPRO, validate_paper_repro_config
from train import TrainConfig as _TC, run_fig9_protocol

_REQUIRED_EXPERIMENTS = {
    "with_nam",
    "without_nam_0",
    "without_nam_2",
    "without_nam_4",
    "without_nam_8",
}


def _validate_protocol(cfg: dict) -> None:
    if cfg.get("channel_type") != "awgn":
        raise RuntimeError("fig9 requires channel_type='awgn'")
    if cfg.get("metrics") != ["bleu1", "bleu2"]:
        raise RuntimeError("fig9 requires metrics=['bleu1','bleu2']")
    if not isinstance(cfg.get("nam_experiments"), dict):
        raise RuntimeError("fig9 requires nam_experiments dict")
    names = set(cfg["nam_experiments"].keys())
    if not _REQUIRED_EXPERIMENTS.issubset(names):
        raise RuntimeError(f"fig9 nam_experiments missing keys: {sorted(_REQUIRED_EXPERIMENTS - names)}")


def _build_train_cfg(cfg: dict, exp_name: str, exp_spec: dict) -> _TC:
    if bool(exp_spec.get("use_nam", False)):
        train_snr_min_db = float(exp_spec["train_snr_min_db"])
        train_snr_max_db = float(exp_spec["train_snr_max_db"])
    else:
        train_snr_min_db = float(exp_spec["train_snr_db"])
        train_snr_max_db = float(exp_spec["train_snr_db"])

    out_dir = Path(cfg["output_dir"]) / exp_name
    ckpt_dir = Path(cfg["checkpoint_dir"]) / exp_name
    cache_dir = Path(cfg["caption_cache_dir"]) / exp_name

    return _TC(
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
        train_snr_min_db=train_snr_min_db,
        train_snr_max_db=train_snr_max_db,
        train_snr_mode=str(exp_spec["train_snr_mode"]),
        use_caption_cache=bool(cfg["train"]["use_caption_cache"]),
        caption_cache_dir=str(cache_dir),
        strict_cache_required=bool(cfg["train"].get("strict_cache_required", True)),
        train_phase_config=cfg["train"].get("train_phase_config"),
        train_tag=f"fig9_{exp_name}",
        fig_name="fig9",
        val_split_ratio=float(cfg.get("val_split_ratio", 0.2)),
        val_split_seed=int(cfg.get("val_split_seed", 42)),
        use_nam=bool(exp_spec.get("use_nam", True)),
        caption_mode=cfg.get("caption_mode", "baseline"),
        caption_prompt=cfg.get("caption_prompt"),
    )


def _build_eval_cfg(cfg: dict, checkpoint_path: str, exp_name: str, use_nam: bool = True) -> _EC:
    out_dir = Path(cfg["output_dir"]) / exp_name
    return _EC(
        project_root=cfg["project_root"],
        model_file=cfg["model_file"],
        target_file=cfg["target_file"],
        test_split_dir=cfg["test_split_dir"],
        output_dir=str(out_dir),
        senders=cfg["senders"],
        blip_ckb_dir=cfg["blip_ckb_dir"],
        ram_ckb_path=cfg["ram_ckb_path"],
        sd_ckb_dir=cfg["sd_ckb_dir"],
        channel_type=cfg["channel_type"],
        ckpt_blip=checkpoint_path,
        ckpt_ram="",
        snr_list=cfg["snr_test_list"],
        metrics=cfg.get("metrics", ["bleu1", "bleu2"]),
        sd_steps=int(cfg["sd_steps"]),
        batch_size=int(cfg["eval"]["batch_size"]),
        max_batches=int(cfg["eval"]["max_batches"]),
        max_per_class=int(cfg["eval"]["max_per_class"]),
        seed=int(cfg["seed"]),
        strict_ckpt=bool(cfg["strict_ckpt"]),
        strict_paper_repro=STRICT_PAPER_REPRO,
        quiet_third_party=bool(cfg["quiet_third_party"]),
        tag=f"{cfg['eval']['tag']}_{exp_name}",
        fig_name="fig9",
        use_nam=bool(use_nam),
        caption_mode=cfg.get("caption_mode", "baseline"),
        caption_prompt=cfg.get("caption_prompt"),
    )


def _extract_primary_checkpoint(train_result: dict) -> str:
    rows = train_result.get("results", [])
    if not rows:
        raise RuntimeError("fig9 train result missing checkpoint rows")
    ckpt = rows[0].get("checkpoint") or rows[0].get("phase_joint_best_checkpoint")
    if not ckpt:
        raise RuntimeError("fig9 train result missing final checkpoint")
    return str(Path(ckpt).resolve())


def _aggregate_eval_curves(output_root: Path, eval_outputs: list[dict]) -> Path:
    out_csv = output_root / "fig9_nam_multi_curve.csv"
    rows = []
    for item in eval_outputs:
        exp_name = item["experiment"]
        curve_csv = Path(item["curve_csv"])
        with curve_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("metric") not in {"bleu1", "bleu2"}:
                    continue
                rows.append(
                    {
                        "experiment": exp_name,
                        "snr_db": row.get("snr_db"),
                        "metric": row.get("metric"),
                        "value": row.get("value"),
                    }
                )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "snr_db", "metric", "value"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_csv


def _has_fig9_best_checkpoint_map(checkpoint_map: dict) -> bool:
    required = ["with_nam", "without_nam_0", "without_nam_2", "without_nam_4", "without_nam_8"]
    for name in required:
        ckpt = Path(str(checkpoint_map.get(name, ""))).expanduser().resolve()
        if not ckpt.exists():
            return False
    return True


def main() -> None:
    args = make_experiment_argparser("fig9").parse_args()

    cfg = build_fig9_config()
    _validate_protocol(cfg)
    validate_paper_repro_config("fig9", cfg)

    # ── smoke / formal 隔离 + 正式准入门卫 ──
    apply_smoke_formal_guard("fig9", cfg)
    inject_provenance(cfg)

    print_experiment_params("fig9", args.mode, cfg)

    checkpoint_map = dict(cfg.get("eval", {}).get("nam_checkpoint_map", {}))
    output_root = Path(cfg["output_dir"])

    if args.mode in {"train", "all"} and bool(cfg.get("train", {}).get("enabled", False)):
        reuse_best = bool(cfg.get("train", {}).get("use_previous_best_checkpoint", False))
        if reuse_best and _has_fig9_best_checkpoint_map(checkpoint_map):
            print(
                "[FIG9] train.use_previous_best_checkpoint=True and NAM checkpoint map is complete; "
                "skip training and reuse existing best models."
            )
        else:
            for exp_name, exp_spec in cfg["nam_experiments"].items():
                print(f"[FIG9_TRAIN] experiment={exp_name}")
                train_cfg = _build_train_cfg(cfg, exp_name=exp_name, exp_spec=exp_spec)
                train_result = run_fig9_protocol(train_cfg)
                checkpoint_map[exp_name] = _extract_primary_checkpoint(train_result)

        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "fig9_nam_checkpoint_map.json").write_text(
            json.dumps(checkpoint_map, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if args.mode in {"eval", "all"} and bool(cfg.get("eval", {}).get("enabled", False)):
        missing = sorted(_REQUIRED_EXPERIMENTS - set(checkpoint_map.keys()))
        if missing:
            raise RuntimeError(f"fig9 missing checkpoint map entries for experiments: {missing}")

        eval_outputs = []
        for exp_name in ["with_nam", "without_nam_0", "without_nam_2", "without_nam_4", "without_nam_8"]:
            print(f"[FIG9_EVAL] experiment={exp_name}")
            exp_use_nam = bool(cfg["nam_experiments"][exp_name].get("use_nam", True))
            eval_cfg = _build_eval_cfg(cfg, checkpoint_path=checkpoint_map[exp_name], exp_name=exp_name, use_nam=exp_use_nam)
            result = run_fig9_eval(eval_cfg)
            result["experiment"] = exp_name
            eval_outputs.append(result)

        multi_curve = _aggregate_eval_curves(output_root, eval_outputs)
        print(f"[FIG9_EVAL] multi_curve_csv={multi_curve}")


if __name__ == "__main__":
    main()
