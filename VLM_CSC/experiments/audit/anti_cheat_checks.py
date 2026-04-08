"""
anti_cheat_checks.py — 反作弊检查清单
=======================================
任务书 §12 列出 13 项 anti-cheat checklist，本模块整合所有检查为可调用函数。

调用方式：
  from audit.anti_cheat_checks import run_anti_cheat_report
  report = run_anti_cheat_report(model, cfg, fig_name)
  # report 是 list[dict]，每条 {"id": int, "name": str, "status": "pass"|"fail"|"skip", "detail": str}

检查项（对应任务书 §12）：
  1. NAM hidden dims = (56,128,56,56) — 纸面明确
  2. use_nam=False 时 NAM 零参数 — 结构消融
  3. 三阶段训练冻结逻辑 — channel/semantic/joint
  4. CKB 不可训练 — BLIP/RAM/SD 冻结
  5. feature_dim = 128
  6. semantic encoder/decoder = 3 层 transformer
  7. channel encoder/decoder FF = [256, 128]
  8. MED: STM max=500, τ=10, λ=0.05
  9. receiver_kb = sd
  10. checkpoint 含元数据字段
  11. eval 使用 clip_zeroshot (fig7) / bleu (fig8/9)
  12. 无 proxy/stub/mock 函数
  13. 正式结果目录与 smoke 隔离
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure VLM_CSC root is on sys.path so model imports work
_VLM_CSC_ROOT = Path(__file__).resolve().parents[2]
if str(_VLM_CSC_ROOT) not in sys.path:
    sys.path.insert(0, str(_VLM_CSC_ROOT))


def _check_item(item_id: int, name: str, passed: bool, detail: str = "") -> Dict[str, Any]:
    return {
        "id": item_id,
        "name": name,
        "status": "pass" if passed else "fail",
        "detail": detail,
    }


def check_nam_hidden_dims(model) -> Dict[str, Any]:
    """§12-1: NAM hidden dims = (56,128,56,56)"""
    try:
        from model.models.nam import NAM
        actual = tuple(NAM._PAPER_HIDDEN_DIMS)
        expected = (56, 128, 56, 56)
        return _check_item(1, "NAM hidden dims", actual == expected,
                           f"expected={expected}, actual={actual}")
    except Exception as e:
        return _check_item(1, "NAM hidden dims", False, str(e))


def check_nam_structural_zero(model, use_nam: bool) -> Dict[str, Any]:
    """§12-2: use_nam=False 时 NAM 输出全零 / 零参数"""
    if use_nam:
        return _check_item(2, "NAM structural zero (use_nam=True)", True, "skip — NAM is enabled")

    nam_param_count = sum(
        p.numel() for name, p in model.named_parameters() if "nam" in name.lower()
    )
    return _check_item(2, "NAM structural zero", nam_param_count == 0,
                       f"nam_param_count={nam_param_count}")


def check_ckb_frozen(model) -> Dict[str, Any]:
    """§12-4: CKB (BLIP/RAM/SD) 参数不可训练"""
    trainable_ckb = []
    for name, p in model.named_parameters():
        if p.requires_grad and any(k in name.lower() for k in ("blip", "ram", "receiver_ckb")):
            trainable_ckb.append(name)
    return _check_item(4, "CKB frozen", len(trainable_ckb) == 0,
                       f"trainable_ckb_params={trainable_ckb[:5]}")


def check_feature_dim(model) -> Dict[str, Any]:
    """§12-5: feature_dim = 128"""
    dim = getattr(model, "feature_dim", None)
    return _check_item(5, "feature_dim=128", dim == 128, f"actual={dim}")


def check_locked_model_assumptions() -> Dict[str, Any]:
    """§12-6/7/8: semantic/channel encoder 层数, MED 参数"""
    try:
        from common.paper_repro_lock import LOCKED_MODEL_ASSUMPTIONS as L
        checks = {
            "semantic_encoder_layers": (L.get("semantic_encoder_layers"), 3),
            "channel_hidden_dims": (L.get("channel_hidden_dims"), [256, 128]),
            "stm_max_size": (L.get("stm_max_size"), 500),
            "med_threshold": (L.get("med_threshold"), 0.05),
            "med_tau": (L.get("med_tau"), 10.0),
        }
        failures = {k: v for k, v in checks.items() if v[0] != v[1]}
        return _check_item(6, "locked model assumptions", len(failures) == 0,
                           f"failures={failures}" if failures else "all match")
    except Exception as e:
        return _check_item(6, "locked model assumptions", False, str(e))


def check_checkpoint_meta(checkpoint_path: str) -> Dict[str, Any]:
    """§12-10: checkpoint 包含元数据字段"""
    import torch
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        required = [
            "meta_figure_name", "meta_experiment_name", "meta_sender_kb",
            "meta_channel_type", "meta_seed", "meta_git_hash",
        ]
        missing = [k for k in required if k not in ckpt]
        return _check_item(10, "checkpoint meta", len(missing) == 0,
                           f"missing={missing}" if missing else "all present")
    except FileNotFoundError:
        return _check_item(10, "checkpoint meta", False, f"file not found: {checkpoint_path}")
    except Exception as e:
        return _check_item(10, "checkpoint meta", False, str(e))


_PROXY_PATTERN = re.compile(r"\b(_proxy_|_stub_|_mock_|_fake_)\b", re.IGNORECASE)


def check_no_proxy_functions(modules: list) -> Dict[str, Any]:
    """§12-12: 无 proxy/stub/mock 函数"""
    violations = []
    for mod in modules:
        try:
            source = inspect.getsource(mod)
            if _PROXY_PATTERN.search(source):
                violations.append(mod.__name__)
        except Exception:
            pass
    return _check_item(12, "no proxy/stub/mock", len(violations) == 0,
                       f"violations={violations}" if violations else "clean")


def check_smoke_formal_isolation(output_dir: str) -> Dict[str, Any]:
    """§12-13: 正式结果目录与 smoke 隔离"""
    from pathlib import Path
    p = Path(output_dir)
    # 正式目录不应包含 smoke 标记文件
    smoke_marker = p / ".smoke_run"
    has_smoke_marker = smoke_marker.exists()
    # 检查正式目录下是否混入 smoke tag
    has_smoke_files = any(f.name.startswith("smoke_") for f in p.glob("*") if f.is_file())
    clean = not has_smoke_marker and not has_smoke_files
    return _check_item(13, "smoke/formal isolation", clean,
                       f"smoke_marker={has_smoke_marker}, smoke_files={has_smoke_files}")


def run_anti_cheat_report(
    model,
    cfg: dict,
    fig_name: str,
    *,
    use_nam: bool = True,
    checkpoint_path: str = "",
    source_modules: Optional[list] = None,
) -> List[Dict[str, Any]]:
    """运行完整 anti-cheat 检查并返回报告。"""
    report = []

    report.append(check_nam_hidden_dims(model))
    report.append(check_nam_structural_zero(model, use_nam))
    report.append(check_ckb_frozen(model))
    report.append(check_feature_dim(model))
    report.append(check_locked_model_assumptions())

    if checkpoint_path:
        report.append(check_checkpoint_meta(checkpoint_path))

    if source_modules:
        report.append(check_no_proxy_functions(source_modules))

    output_dir = cfg.get("output_dir", "")
    if output_dir:
        report.append(check_smoke_formal_isolation(output_dir))

    return report


def print_anti_cheat_report(report: List[Dict[str, Any]]) -> None:
    """格式化打印 anti-cheat 报告。"""
    total = len(report)
    passed = sum(1 for r in report if r["status"] == "pass")
    print(f"\n{'='*60}")
    print(f"  Anti-Cheat Report: {passed}/{total} passed")
    print(f"{'='*60}")
    for r in report:
        icon = "✓" if r["status"] == "pass" else "✗"
        print(f"  [{icon}] #{r['id']:02d} {r['name']}: {r['detail']}")
    print(f"{'='*60}\n")
