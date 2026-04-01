"""Shared bootstrap utilities for all run_figN experiment scripts.

Extracts the repeated audit / smoke-formal isolation / git-hash injection
boilerplate that was duplicated across run_fig7..run_fig10.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP_DIR = Path(__file__).resolve().parent.parent
_AUDIT_DIR = str(_EXP_DIR / "audit")
if _AUDIT_DIR not in sys.path:
    sys.path.insert(0, _AUDIT_DIR)

from formal_guard import run_formal_guard  # noqa: E402
from smoke_formal_isolation import (  # noqa: E402
    is_smoke_run,
    write_smoke_marker,
    assert_no_smoke_in_formal,
    resolve_output_dir,
)
from checkpoint_meta import get_git_hash  # noqa: E402


# ── public re-exports so callers only need one import ──
__all__ = [
    "apply_smoke_formal_guard",
    "inject_provenance",
    "make_experiment_argparser",
    "print_experiment_params",
    "is_smoke_run",
    "resolve_output_dir",
    "get_git_hash",
]


def apply_smoke_formal_guard(
    fig_name: str,
    cfg: dict,
    *,
    output_dirs: list[str] | None = None,
) -> bool:
    """Run the smoke/formal isolation gate and formal guard.

    Parameters
    ----------
    fig_name : str
        e.g. ``"fig7"``
    cfg : dict
        Full experiment config dict.
    output_dirs : list[str] | None
        Override list of dir keys to protect.  Falls back to ``["output_dir"]``
        if not specified.

    Returns
    -------
    bool
        ``True`` if this is a smoke/debug run.
    """
    _smoke = is_smoke_run(cfg)
    dirs = output_dirs or ["output_dir"]

    if _smoke:
        label = fig_name.upper()
        print(f"[{label}-SMOKE] 当前为 smoke/debug 运行，结果不作正式实验用")
        for key in dirs:
            d = cfg.get(key, "")
            if d:
                write_smoke_marker(d)
    else:
        for key in dirs:
            d = cfg.get(key, "")
            if d:
                assert_no_smoke_in_formal(d)
        run_formal_guard(fig_name, cfg)

    return _smoke


def inject_provenance(cfg: dict) -> str:
    """Inject git hash into ``cfg["_provenance"]`` and return it."""
    git_hash = get_git_hash()
    cfg.setdefault("_provenance", {})["git_hash"] = git_hash
    return git_hash


def make_experiment_argparser(
    fig_name: str,
    *,
    extra_args: list[tuple[str, dict]] | None = None,
) -> argparse.ArgumentParser:
    """Create a standard ``--mode train/eval/all`` argument parser.

    Parameters
    ----------
    fig_name : str
        Used in the help description.
    extra_args : list of (flag, kwargs) pairs, optional
        Additional CLI arguments to register.
    """
    parser = argparse.ArgumentParser(description=f"Run {fig_name} experiment")
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    for flag, kwargs in (extra_args or []):
        parser.add_argument(flag, **kwargs)
    return parser


def print_experiment_params(fig_name: str, mode: str, cfg: dict) -> None:
    """在实验启动前显式打印所有关键参数。

    覆盖信道、模型、数据集、训练超参、评估设置、
    caption / SR 模式、输出路径等所有与复现相关的参数。
    """
    SEP = "═" * 62
    sub = "─" * 62

    def _kv(label: str, value: object, width: int = 22) -> str:
        return f"  {label:<{width}}: {value}"

    train_cfg: dict = cfg.get("train", {})
    eval_cfg: dict = cfg.get("eval", {})

    lines: list[str] = []
    lines.append(SEP)
    lines.append(f"  {fig_name.upper()}  模式={mode}  实验参数一览")
    lines.append(sub)

    # ── 模型 / 信道 ───────────────────────────────────────
    lines.append("  [模型 / 信道]")
    lines.append(_kv("channel_type",     cfg.get("channel_type", "—")))
    lines.append(_kv("senders",          cfg.get("senders", "—")))
    lines.append(_kv("channel_dim",      cfg.get("channel_dim", "default")))
    lines.append(_kv("max_text_len",     cfg.get("max_text_len", "default")))
    lines.append(_kv("use_nam",          cfg.get("use_nam", "default")))
    lines.append(sub)

    # ── Caption / SR 模式 ─────────────────────────────────
    _use_ft = bool(cfg.get("use_finetuned_blip", False))
    _ckpt_mode_display = cfg.get("ckpt_mode") or cfg.get("caption_mode", "baseline")
    _raw_prompt = cfg.get("caption_prompt")
    _prompt_display = repr(_raw_prompt) if _raw_prompt is not None else "None  (模式默认: 'a photo of a')"
    lines.append("  [Caption / SR]")
    lines.append(_kv("caption_mode",       cfg.get("caption_mode", "baseline")))
    lines.append(_kv("ckpt_mode",          _ckpt_mode_display))
    lines.append(_kv("use_finetuned_blip", _use_ft))
    lines.append(_kv("caption_prompt",     _prompt_display))
    lines.append(sub)

    # ── 评估参数（eval / all 模式才有意义）────────────────
    if mode in {"eval", "all"}:
        lines.append("  [评估]")
        lines.append(_kv("metrics",          cfg.get("metrics", "—")))
        lines.append(_kv("snr_list",         cfg.get("snr_list") or cfg.get("snr_test_list", "—")))
        lines.append(_kv("sd_steps",         cfg.get("sd_steps", "—")))
        lines.append(_kv("sd_height",        cfg.get("sd_height", 512)))
        lines.append(_kv("sd_width",         cfg.get("sd_width", 512)))
        lines.append(_kv("eval_batch_size",  eval_cfg.get("batch_size", "—")))
        lines.append(_kv("eval_max_batches", eval_cfg.get("max_batches", "—")))
        lines.append(_kv("eval_max_per_cls", eval_cfg.get("max_per_class", "—")))
        # fig8 专有
        if cfg.get("eval_output_mode"):
            lines.append(_kv("eval_output_mode", cfg["eval_output_mode"]))
        if cfg.get("dataset_sequence"):
            lines.append(_kv("dataset_sequence",  cfg["dataset_sequence"]))
        # fig9 专有
        if cfg.get("nam_experiments"):
            exp_names = list(cfg["nam_experiments"].keys())
            lines.append(_kv("nam_experiments",   exp_names))
        # fig10 专有
        if cfg.get("baselines"):
            lines.append(_kv("baselines",         cfg["baselines"]))
        lines.append(_kv("strict_ckpt",       cfg.get("strict_ckpt", "—")))
        lines.append(_kv("required_backend",
                         cfg.get("protocol", {}).get("required_classifier_backend", "—")))
        lines.append(sub)

    # ── 训练参数（train / all 模式才有意义）──────────────
    if mode in {"train", "all"}:
        lines.append("  [训练]")
        snr_min = train_cfg.get("train_snr_min_db", "—")
        snr_max = train_cfg.get("train_snr_max_db", "—")
        snr_mode = train_cfg.get("train_snr_mode", "—")
        lines.append(_kv("train_snr",
                         f"{snr_min} ~ {snr_max}  ({snr_mode})"))
        lines.append(_kv("lr",               train_cfg.get("train_lr", "—")))
        lines.append(_kv("weight_decay",     train_cfg.get("train_weight_decay", 0.0)))
        lines.append(_kv("batch_size",       train_cfg.get("train_batch_size", "—")))
        lines.append(_kv("max_batches",      train_cfg.get("train_max_batches", "—")))
        lines.append(_kv("max_per_class",    train_cfg.get("train_max_per_class", "—")))
        lines.append(_kv("val_ratio",        train_cfg.get("val_ratio", "—")))
        if cfg.get("fig_name") != "fig8":
            phase = train_cfg.get("train_phase_config", {})
            if phase:
                ch_p  = phase.get("channel_phase", {})
                sem_p = phase.get("semantic_phase", {})
                j_p   = phase.get("joint_phase", {})
                ch_ep  = ch_p.get("epochs", "—")
                sem_ep = sem_p.get("epochs", "—")
                j_ep   = j_p.get("max_joint_epochs", "—")
                lines.append(_kv("epochs ch/sem/joint",
                                 f"{ch_ep} / {sem_ep} / {j_ep}"))
                ch_pat  = ch_p.get("early_stop_patience", 0)
                sem_pat = sem_p.get("early_stop_patience", 0)
                j_pat   = j_p.get("early_stop_patience", 0)
                def _pstr(p): return str(p) if p else "off"
                lines.append(_kv("patience ch/sem/joint",
                                 f"{_pstr(ch_pat)} / {_pstr(sem_pat)} / {_pstr(j_pat)}"))
        lines.append(_kv("use_caption_cache", train_cfg.get("use_caption_cache", "—")))
        lines.append(_kv("seed",             cfg.get("seed", "—")))
        lines.append(sub)

    # ── 路径 ─────────────────────────────────────────────
    lines.append("  [路径]")
    if cfg.get("output_dir"):
        lines.append(_kv("output_dir",       cfg["output_dir"]))
    if cfg.get("train_monitor_output_dir"):
        lines.append(_kv("train_output_dir", cfg["train_monitor_output_dir"]))
    if cfg.get("final_eval_output_dir"):
        lines.append(_kv("eval_output_dir",  cfg["final_eval_output_dir"]))
    if cfg.get("checkpoint_dir"):
        lines.append(_kv("checkpoint_dir",   cfg["checkpoint_dir"]))
    if cfg.get("caption_cache_dir"):
        lines.append(_kv("caption_cache_dir", cfg["caption_cache_dir"]))
    lines.append(_kv("model_file",
                     Path(cfg.get("model_file", "—")).name))

    lines.append(SEP)
    print("\n".join(lines))
    print(flush=True)
