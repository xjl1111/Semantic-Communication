"""评估路由：按 fig_name 分发到 _run_evaluation_core。"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict

from eval.config import EvalConfig
from eval.core import _run_evaluation_core


def run_fig7_eval(config: EvalConfig) -> Dict[str, str]:
    cfg = replace(config, fig_name="fig7")
    return _run_evaluation_core(cfg)


def run_fig8_continual_evaluation(config: EvalConfig) -> Dict[str, str]:
    cfg = replace(config, fig_name="fig8")
    return _run_evaluation_core(cfg)


def run_fig9_eval(config: EvalConfig) -> Dict[str, str]:
    cfg = replace(config, fig_name="fig9")
    return _run_evaluation_core(cfg)


def run_fig10_baseline_evaluation(config: EvalConfig) -> Dict[str, str]:
    cfg = replace(config, fig_name="fig10")
    return _run_evaluation_core(cfg)


def run_evaluation(config: EvalConfig) -> Dict[str, str]:
    fig_name = str(config.fig_name).strip().lower()
    if fig_name == "fig8":
        return run_fig8_continual_evaluation(config)
    if fig_name == "fig9":
        return run_fig9_eval(config)
    if fig_name == "fig10":
        return run_fig10_baseline_evaluation(config)
    return run_fig7_eval(config)
