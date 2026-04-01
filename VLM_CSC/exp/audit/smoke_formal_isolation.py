"""
smoke_formal_isolation.py — Smoke / Formal 目录隔离
=====================================================
任务书 §10 / §18 / §33 要求：
  - smoke (调试级) 结果写入 ``data/experiments/figN/smoke/``
  - formal (正式)  结果写入 ``data/experiments/figN/``
  - 两者不得混合，正式目录不允许出现 smoke 标记

本模块提供：
  - is_smoke_run(cfg): 判断当前运行是否为 smoke 级
  - resolve_output_dir(cfg, fig_name): 返回正确的输出目录
  - write_smoke_marker(output_dir): 在 smoke 目录写入标记文件
  - assert_no_smoke_in_formal(output_dir): 阻断正式目录中的 smoke 污染
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def is_smoke_run(cfg: dict) -> bool:
    """根据训练配置判断是否为 smoke/debug 级别运行。

    判定标准（满足任一即为 smoke）：
      - train_max_per_class > 0 且 < 100
      - train_max_batches > 0 且 < 10
      - train_phase_config 各 phase epoch 总和 < 3
    """
    train_cfg = cfg.get("train", {})

    max_pc = int(train_cfg.get("train_max_per_class", -1))
    if 0 < max_pc < 100:
        return True

    max_batches = int(train_cfg.get("train_max_batches", -1))
    if 0 < max_batches < 10:
        return True

    # 检查真正使用的 phase epoch 总和（而非未使用的 train_epochs 字段）
    phase_cfg = train_cfg.get("train_phase_config") or {}
    total_phase_epochs = (
        int(phase_cfg.get("channel_phase", {}).get("epochs", 0))
        + int(phase_cfg.get("semantic_phase", {}).get("epochs", 0))
        + int(phase_cfg.get("joint_phase", {}).get("max_joint_epochs", 0))
    )
    if total_phase_epochs < 3:
        return True

    return False


def resolve_output_dir(base_output_dir: str, fig_name: str, cfg: dict) -> str:
    """根据 smoke/formal 状态返回正确的输出目录。

    Parameters
    ----------
    base_output_dir : str
        figure 级基础输出目录 (e.g. data/experiments/fig7)
    fig_name : str
        "fig7" / "fig8" / ...
    cfg : dict
        完整配置

    Returns
    -------
    str
        smoke 运行返回 ``base/smoke/``，正式运行返回 ``base/``
    """
    base = Path(base_output_dir)
    if is_smoke_run(cfg):
        return str(base / "smoke")
    return str(base)


def write_smoke_marker(output_dir: str) -> Path:
    """在输出目录写入 ``.smoke_run`` 标记文件。"""
    marker = Path(output_dir) / ".smoke_run"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        f"This directory contains SMOKE/DEBUG results.\n"
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n"
        f"These results MUST NOT be used as formal experiment outputs.\n",
        encoding="utf-8",
    )
    return marker


def assert_no_smoke_in_formal(output_dir: str) -> None:
    """正式运行前检查输出目录是否被 smoke 结果污染。"""
    p = Path(output_dir)
    if not p.exists():
        return  # 全新目录，无需检查

    smoke_marker = p / ".smoke_run"
    if smoke_marker.exists():
        raise RuntimeError(
            f"正式实验输出目录被 smoke 结果污染: {output_dir}\n"
            f"请清理此目录或使用独立的正式输出目录。"
        )

    # 检查是否有明显的 smoke 产物
    smoke_files = [f for f in p.glob("smoke_*") if f.is_file()]
    if smoke_files:
        raise RuntimeError(
            f"正式实验输出目录包含 smoke 标记文件: {[f.name for f in smoke_files[:5]]}\n"
            f"请清理此目录。"
        )
