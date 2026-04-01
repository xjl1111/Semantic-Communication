"""
checkpoint_meta.py — 检查点元数据生成
======================================
任务书 §8.3 要求检查点必须包含以下字段：
  figure_name, experiment_name, dataset_split_id,
  sender_kb, receiver_kb, use_med, use_nam,
  channel_type, train_snr_mode, seed, git_hash, config_hash

本模块提供：
  - build_checkpoint_meta(): 从训练配置构建元数据 dict
  - enrich_checkpoint_dict(): 向现有 torch.save dict 追加元数据
  - get_git_hash(): 获取当前仓库 git commit hash
  - get_config_hash(): 对配置 dict 取 SHA-256 指纹
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def get_git_hash(repo_root: Optional[str] = None) -> str:
    """返回当前仓库的短 commit hash，取不到则返回 'unknown'。"""
    try:
        cwd = repo_root or str(Path(__file__).resolve().parents[2])
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=cwd, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_config_hash(cfg: dict) -> str:
    """对配置 dict 取 SHA-256 前16位作为指纹。"""
    payload = json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_checkpoint_meta(
    *,
    figure_name: str,
    experiment_name: str,
    dataset_split_id: str,
    sender_kb: str,
    receiver_kb: str = "sd",
    use_med: bool = False,
    use_nam: bool = True,
    channel_type: str,
    train_snr_mode: str,
    seed: int,
    phase: str,
    config_dict: Optional[dict] = None,
    repo_root: Optional[str] = None,
) -> Dict[str, Any]:
    """构建符合任务书 §8.3 的检查点元数据。

    Returns
    -------
    dict
        可直接合并到 torch.save 的 state dict 中。
    """
    return {
        "meta_figure_name": figure_name,
        "meta_experiment_name": experiment_name,
        "meta_dataset_split_id": dataset_split_id,
        "meta_sender_kb": sender_kb,
        "meta_receiver_kb": receiver_kb,
        "meta_use_med": use_med,
        "meta_use_nam": use_nam,
        "meta_channel_type": channel_type,
        "meta_train_snr_mode": train_snr_mode,
        "meta_seed": seed,
        "meta_phase": phase,
        "meta_git_hash": get_git_hash(repo_root),
        "meta_config_hash": get_config_hash(config_dict) if config_dict else "n/a",
    }


def enrich_checkpoint_dict(
    ckpt_dict: dict,
    *,
    figure_name: str,
    experiment_name: str,
    dataset_split_id: str,
    sender_kb: str,
    receiver_kb: str = "sd",
    use_med: bool = False,
    use_nam: bool = True,
    channel_type: str,
    train_snr_mode: str,
    seed: int,
    phase: str,
    config_dict: Optional[dict] = None,
    repo_root: Optional[str] = None,
) -> dict:
    """向已有 checkpoint dict 注入任务书要求的元数据字段，原地修改并返回。"""
    meta = build_checkpoint_meta(
        figure_name=figure_name,
        experiment_name=experiment_name,
        dataset_split_id=dataset_split_id,
        sender_kb=sender_kb,
        receiver_kb=receiver_kb,
        use_med=use_med,
        use_nam=use_nam,
        channel_type=channel_type,
        train_snr_mode=train_snr_mode,
        seed=seed,
        phase=phase,
        config_dict=config_dict,
        repo_root=repo_root,
    )
    ckpt_dict.update(meta)
    return ckpt_dict
