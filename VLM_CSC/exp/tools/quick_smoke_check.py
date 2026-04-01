from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable
import sys

_THIS_DIR = Path(__file__).resolve().parent
_EXP_DIR = _THIS_DIR.parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from common.paper_repro_lock import validate_paper_repro_config
from fig7.fig7_config import build_fig7_config
from fig8.fig8_config import build_fig8_config
from fig9.fig9_config import build_fig9_config
from fig10.fig10_config import build_fig10_config


def _validate_fig7_protocol_local(cfg: dict) -> None:
    protocol = cfg.get("protocol", {})
    if protocol.get("locked", False):
        if cfg.get("senders") != ["blip", "ram"]:
            raise RuntimeError("fig7 senders must be ['blip','ram']")
        if cfg.get("metrics") != ["ssq"]:
            raise RuntimeError("fig7 metrics must be ['ssq']")
        if not bool(cfg.get("strict_ckpt")):
            raise RuntimeError("fig7 strict_ckpt must be enabled")
        if protocol.get("receiver_kb") != "sd":
            raise RuntimeError("fig7 receiver_kb must be 'sd'")


def _validate_fig8_protocol_local(cfg: dict) -> None:
    if cfg.get("channel_type") != "rayleigh":
        raise RuntimeError("fig8 channel_type must be rayleigh")
    if cfg.get("metrics") != ["bleu1", "bleu2"]:
        raise RuntimeError("fig8 metrics must be ['bleu1','bleu2']")
    if cfg.get("eval_output_mode") != "continual_learning_map":
        raise RuntimeError("fig8 eval_output_mode must be continual_learning_map")
    if cfg.get("dataset_sequence") != ["cifar", "birds", "catsvsdogs"]:
        raise RuntimeError("fig8 dataset_sequence mismatch")
    med_variants = cfg.get("med_variants")
    if med_variants not in (["with_med", "without_med"], [True, False]):
        raise RuntimeError("fig8 med_variants mismatch")


def _resolve_fig8_checkpoint_map_local(cfg: dict) -> dict:
    raw_map = cfg.get("eval", {}).get("fig8_variant_checkpoint_map")
    if not isinstance(raw_map, dict):
        raise RuntimeError("fig8 missing eval.fig8_variant_checkpoint_map")
    required_variants = ["with_med", "without_med"]
    required_senders = ["blip", "ram"]
    required_tasks = ["cifar", "birds", "catsvsdogs"]
    for variant in required_variants:
        if variant not in raw_map or not isinstance(raw_map[variant], dict):
            raise RuntimeError(f"fig8 checkpoint map missing variant: {variant}")
        for sender in required_senders:
            sender_block = raw_map[variant].get(sender)
            if not isinstance(sender_block, dict):
                raise RuntimeError(f"fig8 checkpoint map missing sender={sender} in variant={variant}")
            for task in required_tasks:
                ckpt = sender_block.get(task)
                if not ckpt:
                    raise RuntimeError(f"fig8 checkpoint map missing task={task} in {variant}/{sender}")
                p = Path(str(ckpt)).expanduser().resolve()
                if not p.exists():
                    raise RuntimeError(f"fig8 checkpoint not found: {p}")
    return raw_map


def _validate_fig9_protocol_local(cfg: dict) -> None:
    if cfg.get("channel_type") != "awgn":
        raise RuntimeError("fig9 channel_type must be awgn")
    if cfg.get("metrics") != ["bleu1", "bleu2"]:
        raise RuntimeError("fig9 metrics must be ['bleu1','bleu2']")
    if not isinstance(cfg.get("nam_experiments"), dict):
        raise RuntimeError("fig9 nam_experiments must be a dict")


def _validate_fig10_protocol_local(cfg: dict) -> None:
    if cfg.get("channel_type") != "awgn":
        raise RuntimeError("fig10 channel_type must be awgn")
    if cfg.get("baselines") != ["vlm_csc", "jscc", "witt"]:
        raise RuntimeError("fig10 baselines mismatch")
    if cfg.get("metrics") != ["classification_accuracy", "compression_ratio", "trainable_parameters"]:
        raise RuntimeError("fig10 metrics mismatch")


def _count_images(root: Path) -> int:
    if not root.exists():
        return 0
    count = 0
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        count += len(list(root.rglob(ext)))
    return count


def _assert_split_has_images(split_path: str, name: str) -> None:
    p = Path(split_path).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f"{name} split dir not found: {p}")
    if _count_images(p) <= 0:
        raise RuntimeError(f"{name} split has no images: {p}")


def _assert_paths_exist(paths: Iterable[str], title: str) -> None:
    missing = []
    for x in paths:
        p = Path(str(x)).expanduser().resolve()
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise RuntimeError(f"{title} missing files: {missing}")


def smoke_fig7() -> str:
    cfg = build_fig7_config()
    _validate_fig7_protocol_local(cfg)
    validate_paper_repro_config("fig7", cfg)

    _assert_split_has_images(cfg["train_split_dir"], "fig7 train")
    _assert_split_has_images(cfg["test_split_dir"], "fig7 test")
    _assert_paths_exist([cfg["eval"]["ckpt_blip"], cfg["eval"]["ckpt_ram"]], "fig7 checkpoints")
    return "fig7 smoke PASS"


def smoke_fig8() -> str:
    cfg = build_fig8_config()
    _validate_fig8_protocol_local(cfg)
    validate_paper_repro_config("fig8", cfg)

    for task in cfg["dataset_sequence"]:
        split_cfg = cfg["dataset_splits"][task]
        _assert_split_has_images(split_cfg["train"], f"fig8 {task} train")
        _assert_split_has_images(split_cfg["test"], f"fig8 {task} test")

    resolved = _resolve_fig8_checkpoint_map_local(cfg)
    total = 0
    for variant_block in resolved.values():
        for sender_block in variant_block.values():
            total += len(sender_block)
    if total != 12:
        raise RuntimeError(f"fig8 checkpoint map size mismatch, expected 12, got {total}")
    return "fig8 smoke PASS"


def smoke_fig9() -> str:
    cfg = build_fig9_config()
    _validate_fig9_protocol_local(cfg)
    validate_paper_repro_config("fig9", cfg)

    _assert_split_has_images(cfg["train_split_dir"], "fig9 train")
    _assert_split_has_images(cfg["test_split_dir"], "fig9 test")

    ckpt_map = cfg.get("eval", {}).get("nam_checkpoint_map", {})
    required = ["with_nam", "without_nam_0", "without_nam_2", "without_nam_4", "without_nam_8"]
    missing_keys = [k for k in required if k not in ckpt_map]
    if missing_keys:
        raise RuntimeError(f"fig9 checkpoint map missing keys: {missing_keys}")
    _assert_paths_exist([ckpt_map[k] for k in required], "fig9 checkpoints")
    return "fig9 smoke PASS"


def smoke_fig10() -> str:
    cfg = build_fig10_config()
    _validate_fig10_protocol_local(cfg)
    validate_paper_repro_config("fig10", cfg)

    _assert_split_has_images(cfg["train_split_dir"], "fig10 train")
    _assert_split_has_images(cfg["test_split_dir"], "fig10 test")

    baseline_ckpts = cfg.get("eval", {}).get("baseline_checkpoints", {})
    required = ["vlm_csc", "jscc", "witt"]
    missing_keys = [k for k in required if k not in baseline_ckpts]
    if missing_keys:
        raise RuntimeError(f"fig10 baseline_checkpoints missing keys: {missing_keys}")
    _assert_paths_exist([baseline_ckpts[k] for k in required], "fig10 baseline checkpoints")
    return "fig10 smoke PASS"


def main() -> None:
    parser = argparse.ArgumentParser(description="30s-level quick smoke checks for fig7/8/9/10")
    parser.add_argument("--fig", choices=["fig7", "fig8", "fig9", "fig10", "all"], default="all")
    args = parser.parse_args()

    checks = {
        "fig7": smoke_fig7,
        "fig8": smoke_fig8,
        "fig9": smoke_fig9,
        "fig10": smoke_fig10,
    }

    selected = [args.fig] if args.fig != "all" else ["fig7", "fig8", "fig9", "fig10"]

    t0 = time.perf_counter()
    for name in selected:
        s0 = time.perf_counter()
        msg = checks[name]()
        dt = time.perf_counter() - s0
        print(f"[{name}] {msg} ({dt:.2f}s)")

    total = time.perf_counter() - t0
    print(f"[quick-smoke] done in {total:.2f}s")


if __name__ == "__main__":
    main()
