"""
formal_guard.py — 正式实验准入门卫
====================================
任务书 §10.4 / §8.3 要求：正式实验之前必须拦截以下情况
  1. allow_fallback = True（不允许降级旁路）
  2. allow_mock = True（不允许 mock 替代真实组件）
  3. 存在 _proxy_ / _stub_ / _mock_ 函数/方法（源码级审查）
  4. 缺少必需的训练检查点
  5. 缺少真实标签（cat/dog 二分类）
  6. 训练样本量过低（smoke 级别参数）
  7. 数据集目录不存在或为空

在每个 run_figN.py 的 main() 入口处调用 ``run_formal_guard(fig_name, cfg)``。
只有全部检查通过才放行。
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


class FormalGuardError(RuntimeError):
    """正式实验准入条件不满足时抛出。"""
    pass


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------

def run_formal_guard(fig_name: str, cfg: dict, *, source_modules: Optional[List] = None) -> None:
    """对一个 figure 配置执行全部正式准入检查。

    Parameters
    ----------
    fig_name : str
        "fig7" / "fig8" / "fig9" / "fig10"
    cfg : dict
        figure 级完整配置（与 run_figN.py 中一致）
    source_modules : list, optional
        需要做源码级 proxy/stub/mock 扫描的 Python 模块对象。
        缺省时仅做配置级检查。
    """
    errors: List[str] = []

    # 1. allow_fallback / allow_mock 必须为 False 或不存在
    _check_no_fallback_mock(cfg, errors)

    # 2. 训练样本量不得过低
    _check_sample_size(fig_name, cfg, errors)

    # 3. 数据集目录必须存在且非空
    _check_dataset_dirs(fig_name, cfg, errors)

    # 4. checkpoint 存在性（eval 模式需要）
    _check_checkpoints(fig_name, cfg, errors)

    # 5. 真实标签可用性
    _check_real_labels(fig_name, cfg, errors)

    # 6. 源码级 proxy / stub / mock 检测
    if source_modules:
        _check_proxy_functions(source_modules, errors)

    if errors:
        msg = f"[FORMAL_GUARD] {fig_name} 正式实验准入失败 ({len(errors)} 项):\n"
        for i, e in enumerate(errors, 1):
            msg += f"  [{i}] {e}\n"
        raise FormalGuardError(msg)


# ---------------------------------------------------------------------------
# 检查函数
# ---------------------------------------------------------------------------

def _check_no_fallback_mock(cfg: dict, errors: List[str]) -> None:
    """递归扫描 cfg dict，拒绝 allow_fallback=True / allow_mock=True。"""
    _scan_dict_for_flags(cfg, path="cfg", errors=errors)


def _scan_dict_for_flags(d: dict, path: str, errors: List[str]) -> None:
    for key, val in d.items():
        current_path = f"{path}.{key}"
        if key == "allow_fallback" and val is True:
            errors.append(f"{current_path}=True — 正式实验不允许 fallback 降级")
        if key == "allow_mock" and val is True:
            errors.append(f"{current_path}=True — 正式实验不允许 mock 替代")
        if isinstance(val, dict):
            _scan_dict_for_flags(val, current_path, errors)


def _check_sample_size(fig_name: str, cfg: dict, errors: List[str]) -> None:
    """训练样本量检查：train_max_per_class == -1（全量）或 >= 100。"""
    train_cfg = cfg.get("train", {})
    max_pc = int(train_cfg.get("train_max_per_class", -1))
    if 0 < max_pc < 100:
        errors.append(
            f"train_max_per_class={max_pc} 过小 — 正式实验要求 -1（全量）或 >=100"
        )

    max_batches = int(train_cfg.get("train_max_batches", -1))
    if 0 < max_batches < 10:
        errors.append(
            f"train_max_batches={max_batches} 过小 — 正式实验要求 -1（全量）或 >=10"
        )


def _check_dataset_dirs(fig_name: str, cfg: dict, errors: List[str]) -> None:
    """检查数据集目录存在且包含图片。"""
    dirs_to_check: List[str] = []

    train_dir = cfg.get("train_split_dir", "")
    if train_dir:
        dirs_to_check.append(train_dir)

    test_dir = cfg.get("test_split_dir", "")
    if test_dir:
        dirs_to_check.append(test_dir)

    # fig8 多数据集
    dataset_roots = cfg.get("dataset_roots", {})
    if isinstance(dataset_roots, dict):
        for name, root in dataset_roots.items():
            dirs_to_check.append(root)

    for d in dirs_to_check:
        p = Path(d)
        if not p.exists():
            errors.append(f"数据集目录不存在: {d}")
        elif p.is_dir() and not any(p.rglob("*.jpg")) and not any(p.rglob("*.png")) and not any(p.rglob("*.jpeg")):
            errors.append(f"数据集目录无图片文件: {d}")


def _check_checkpoints(fig_name: str, cfg: dict, errors: List[str]) -> None:
    """eval 模式下必须有训练好的 checkpoint。"""
    eval_cfg = cfg.get("eval", {})
    if not eval_cfg.get("enabled", False):
        return

    # 如果训练也已启用，训练阶段会在 eval 前产出 checkpoint，跳过检查
    if bool(cfg.get("train", {}).get("enabled", False)):
        return

    if fig_name == "fig7":
        for key in ("ckpt_blip", "ckpt_ram"):
            ckpt = str(eval_cfg.get(key, ""))
            if ckpt and not Path(ckpt).expanduser().resolve().exists():
                errors.append(f"eval 模式缺少 checkpoint: {key}={ckpt}")

    elif fig_name == "fig8":
        ckpt_map = eval_cfg.get("fig8_variant_checkpoint_map", {})
        if not isinstance(ckpt_map, dict) or not ckpt_map:
            errors.append("fig8 eval 模式缺少 fig8_variant_checkpoint_map")

    elif fig_name == "fig9":
        ckpt_map = eval_cfg.get("nam_checkpoint_map", {})
        if not isinstance(ckpt_map, dict) or not ckpt_map:
            errors.append("fig9 eval 模式缺少 nam_checkpoint_map")

    elif fig_name == "fig10":
        baseline_ckpts = eval_cfg.get("baseline_checkpoints", {})
        if not isinstance(baseline_ckpts, dict) or "vlm_csc" not in baseline_ckpts:
            errors.append("fig10 eval 模式缺少 baseline_checkpoints.vlm_csc")


def _check_real_labels(fig_name: str, cfg: dict, errors: List[str]) -> None:
    """确保使用真实标签路径（cat/dog 子目录结构），而不是合成标签。"""
    test_dir = cfg.get("test_split_dir", "")
    if not test_dir:
        return
    test_p = Path(test_dir)
    if not test_p.exists():
        return  # 由 _check_dataset_dirs 报告
    # CatsvsDogs 分类任务需要 cat/ dog/ 子目录
    if fig_name in ("fig7", "fig10"):
        expected_classes = {"cat", "dog"}
        actual = {d.name for d in test_p.iterdir() if d.is_dir()}
        if not expected_classes.issubset(actual):
            errors.append(
                f"test_split_dir 缺少分类子目录: 期望 {expected_classes}, 实际 {actual}"
            )


_PROXY_PATTERN = re.compile(r"\b(_proxy_|_stub_|_mock_|_fake_)", re.IGNORECASE)


def _check_proxy_functions(modules: list, errors: List[str]) -> None:
    """对加载的模块做源码级扫描，拒绝包含 _proxy_ / _stub_ / _mock_ 函数名。"""
    for mod in modules:
        try:
            source = inspect.getsource(mod)
        except (TypeError, OSError):
            continue
        matches = _PROXY_PATTERN.findall(source)
        if matches:
            errors.append(
                f"模块 {mod.__name__} 包含禁止的函数名模式: {sorted(set(matches))}"
            )
