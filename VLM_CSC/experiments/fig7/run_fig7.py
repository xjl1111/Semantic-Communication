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

from eval import EvalConfig, run_evaluation
from common.experiment_bootstrap import (
    apply_smoke_formal_guard,
    inject_provenance,
    make_experiment_argparser,
    print_experiment_params,
)

from fig7_config import build_fig7_config

# audit 子包中的辅助函数
import importlib as _il, types as _ty
def _load_audit(mod_name: str) -> _ty.ModuleType:
    """从 audit/ 子目录加载模块，兼容 Pylance 静态分析。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(mod_name, str(_EXP_DIR / "audit" / f"{mod_name}.py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m

_ckpt_meta = _load_audit("checkpoint_meta")
_smoke_iso = _load_audit("smoke_formal_isolation")
get_git_hash = _ckpt_meta.get_git_hash
is_smoke_run = _smoke_iso.is_smoke_run
resolve_output_dir = _smoke_iso.resolve_output_dir
from common.paper_repro_lock import STRICT_PAPER_REPRO, validate_paper_repro_config
from train import TrainConfig, run_training


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║           ★  命 令 行 参 数 定 义 （集中管理）  ★                     ║
# ║  所有可通过命令行覆盖的参数集中定义在此处。                            ║
# ║  格式：("--flag", {argparse kwargs})                                   ║
# ║  运行时这些值会覆盖 fig7_config.py 中的默认值。                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

CLI_ARGS: list[tuple[str, dict]] = [
    # ─── 评估参数 ─────────────────────────────────────────────────────────
    ("--sd_steps",       {"type": int,   "default": None,
                          "help": "DDIM 去噪步数（覆盖配置）"}),
    ("--sd_height",      {"type": int,   "default": None,
                          "help": "SD 生成图像高度"}),
    ("--sd_width",       {"type": int,   "default": None,
                          "help": "SD 生成图像宽度"}),
    ("--max_per_class",  {"type": int,   "default": None,
                          "help": "评估时每类最大样本数（-1=全部）"}),
    ("--max_batches",    {"type": int,   "default": None,
                          "help": "评估时最大 batch 数（-1=全部）"}),
    ("--batch_size",     {"type": int,   "default": None,
                          "help": "评估/训练 batch 大小"}),
    ("--tag",            {"type": str,   "default": None,
                          "help": "实验标签（用于区分不同运行）"}),
    # ─── 训练参数 ─────────────────────────────────────────────────────────
    ("--train_epochs",         {"type": int, "default": None,
                                "help": "训练 epoch 数"}),
    ("--train_max_per_class",  {"type": int, "default": None,
                                "help": "训练时每类最大样本数（-1=全部）"}),
    # ─── Caption 模式 ────────────────────────────────────────────────────
    ("--caption_mode",   {"type": str,   "default": None,
                          "choices": ["sr", "sr_prompt", "prompt", "blip2"],
                          "help": "BLIP caption 模式: sr / sr_prompt / prompt / blip2"}),
    # ─── CLIP 分类器微调 ────────────────────────────────────────────────
    ("--finetune_clip",  {"action": "store_true",
                          "help": "评估前先微调 CLIP 分类器（linear probe）"}),
    # ─── 协议 ────────────────────────────────────────────────────────────
    ("--allow_protocol_override", {"action": "store_true",
                                   "help": "允许覆盖协议锁定参数（仅限调试）"}),
]


# ══════════════════════════════════════════════════════════════════════════
#  内部工具函数
# ══════════════════════════════════════════════════════════════════════════

def _validate_fig7_protocol(cfg: dict, mode: str) -> None:
    protocol = cfg.get("protocol", {})
    if not protocol.get("locked", False):
        return

    required_senders = protocol.get("required_senders", [])
    required_metrics = protocol.get("required_metrics", [])
    if cfg.get("senders") != required_senders:
        raise RuntimeError(f"Fig7 protocol violation: senders must be {required_senders}, got {cfg.get('senders')}")
    if cfg.get("metrics") != required_metrics:
        raise RuntimeError(f"Fig7 protocol violation: metrics must be {required_metrics}, got {cfg.get('metrics')}")
    if bool(cfg.get("strict_ckpt")) != bool(protocol.get("require_strict_ckpt", True)):
        raise RuntimeError("Fig7 protocol violation: strict_ckpt must remain enabled.")
    if cfg.get("protocol", {}).get("receiver_kb") != "sd":
        raise RuntimeError("Fig7 protocol violation: receiver-side KB must be SD.")

    if mode in {"eval", "all"}:
        if not cfg.get("eval", {}).get("ckpt_blip"):
            raise RuntimeError("Fig7 protocol violation: eval checkpoint for BLIP must be configured.")
        if "ram" in cfg.get("senders", []) and not cfg.get("eval", {}).get("ckpt_ram"):
            raise RuntimeError("Fig7 protocol violation: eval checkpoint for RAM must be configured (ENABLE_RAM=True).")


def _snapshot_config(cfg: dict, mode: str) -> Path:
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_tag = cfg.get("eval", {}).get("tag", "eval")
    train_tag = cfg.get("train", {}).get("train_tag", "train")
    run_tag = eval_tag if mode in {"eval", "all"} else train_tag

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "protocol_name": cfg.get("protocol", {}).get("name", "unknown"),
        "git_hash": cfg.get("_provenance", {}).get("git_hash", get_git_hash()),
        "config": cfg,
    }
    protocol_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "senders": cfg.get("senders"),
                "metrics": cfg.get("metrics"),
                "snr_list": cfg.get("snr_list"),
                "sd_steps": cfg.get("sd_steps"),
                "strict_ckpt": cfg.get("strict_ckpt"),
                "required_classifier_backend": cfg.get("protocol", {}).get("required_classifier_backend"),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    payload["protocol_fingerprint_sha256"] = protocol_fingerprint

    snapshot_path = output_dir / f"fig7_{run_tag}_config_snapshot.json"
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot_path


def _run_clip_finetune(cfg: dict) -> None:
    """执行 CLIP 分类器微调（linear probe on frozen CLIP）。"""
    from finetune_clip_fig7 import main as _clip_ft_main
    _clip_ft_main(
        data_root=cfg["train_split_dir"],
        output_path=cfg["clip_classifier_path"],
        epochs=cfg.get("clip_ft_epochs", 10),
        lr=cfg.get("clip_ft_lr", 1e-3),
        batch_size=cfg.get("clip_ft_batch_size", 32),
    )


def _build_fig7_train_config(cfg: dict) -> TrainConfig:
    return TrainConfig(
        project_root=cfg["project_root"],
        model_file=cfg["model_file"],
        train_split_dir=cfg["train_split_dir"],
        output_dir=cfg["output_dir"],
        checkpoint_dir=cfg["checkpoint_dir"],
        senders=cfg["senders"],
        blip_ckb_dir=cfg["blip_ckb_dir"],
        ram_ckb_path=cfg.get("ram_ckb_path", ""),
        sd_ckb_dir=cfg["sd_ckb_dir"],
        channel_type=cfg["channel_type"],
        seed=cfg["seed"],
        quiet_third_party=cfg["quiet_third_party"],
        strict_paper_repro=STRICT_PAPER_REPRO,
        max_text_len=int(cfg.get("max_text_len", 24)),
        max_text_len_by_sender=cfg.get("max_text_len_by_sender"),
        train_epochs=int(cfg["train"].get("train_epochs", 1)),
        train_lr=cfg["train"]["train_lr"],
        train_weight_decay=cfg["train"]["train_weight_decay"],
        train_batch_size=cfg["train"]["train_batch_size"],
        train_max_batches=cfg["train"]["train_max_batches"],
        train_max_per_class=cfg["train"]["train_max_per_class"],
        val_ratio=cfg["train"]["val_ratio"],
        train_snr_min_db=cfg["train"]["train_snr_min_db"],
        train_snr_max_db=cfg["train"]["train_snr_max_db"],
        train_snr_mode=cfg["train"]["train_snr_mode"],
        use_caption_cache=bool(cfg["train"]["use_caption_cache"]),
        strict_cache_required=bool(cfg["train"].get("strict_cache_required", True)),
        train_phase_config=cfg["train"].get("train_phase_config"),
        caption_cache_dir=cfg["caption_cache_dir"],
        train_tag=cfg["train"]["train_tag"],
        channel_dim=cfg.get("channel_dim"),
        caption_mode=cfg.get("caption_mode", "baseline"),
        caption_prompt=cfg.get("caption_prompt"),
    )


def _resolve_fig7_eval_checkpoints(cfg: dict) -> None:
    eval_cfg = cfg.setdefault("eval", {})
    blip_ckpt = Path(str(eval_cfg.get("ckpt_blip", ""))).expanduser().resolve()
    ram_ckpt = Path(str(eval_cfg.get("ckpt_ram", ""))).expanduser().resolve()

    ram_ckpt = Path(str(eval_cfg.get("ckpt_ram", ""))).expanduser().resolve()
    need_ram = "ram" in cfg.get("senders", [])

    all_present = blip_ckpt.exists() and (not need_ram or ram_ckpt.exists())
    if all_present:
        return

    if not bool(cfg.get("train", {}).get("enabled", False)):
        raise RuntimeError(
            "Fig7 eval checkpoints are missing and train.enabled is False; cannot bootstrap checkpoints."
        )

    print("[FIG7] eval checkpoints missing, running training bootstrap...")
    train_result = run_training(_build_fig7_train_config(cfg))
    rows = train_result.get("results")
    rows = list(rows) if isinstance(rows, list) else []
    sender_ckpt_map = {str(row.get("sender")): str(row.get("checkpoint")) for row in rows if row.get("sender") and row.get("checkpoint")}

    required = set(cfg.get("senders", []))
    missing = required - set(sender_ckpt_map.keys())
    if missing:
        raise RuntimeError(
            f"Fig7 training bootstrap did not produce checkpoints for: {sorted(missing)}, got senders={sorted(sender_ckpt_map.keys())}"
        )

    eval_cfg["ckpt_blip"] = sender_ckpt_map["blip"]
    if need_ram:
        eval_cfg["ckpt_ram"] = sender_ckpt_map["ram"]
    print(f"[FIG7] bootstrap checkpoints: {sender_ckpt_map}")


def main() -> None:
    args = make_experiment_argparser("fig7", extra_args=CLI_ARGS).parse_args()

    cfg = build_fig7_config()
    protocol_locked = bool(cfg.get("protocol", {}).get("locked", False))
    protocol_override_fields = []

    if args.sd_steps is not None:
        cfg["sd_steps"] = args.sd_steps
        protocol_override_fields.append("sd_steps")
    if args.sd_height is not None:
        cfg["sd_height"] = args.sd_height
        protocol_override_fields.append("sd_height")
    if args.sd_width is not None:
        cfg["sd_width"] = args.sd_width
        protocol_override_fields.append("sd_width")
    if args.max_per_class is not None:
        cfg["eval"]["max_per_class"] = args.max_per_class
        # max_per_class is a sampling control, not a protocol-level parameter
    if args.max_batches is not None:
        cfg["eval"]["max_batches"] = args.max_batches
        protocol_override_fields.append("eval.max_batches")
    if args.batch_size is not None:
        cfg["eval"]["batch_size"] = args.batch_size
        cfg["train"]["train_batch_size"] = args.batch_size
        protocol_override_fields.append("eval.batch_size")
        protocol_override_fields.append("train.train_batch_size")
    if args.tag is not None:
        cfg["eval"]["tag"] = args.tag
        cfg["train"]["train_tag"] = args.tag
    if args.train_epochs is not None:
        cfg["train"]["train_epochs"] = args.train_epochs
        protocol_override_fields.append("train.train_epochs")
    if args.train_max_per_class is not None:
        cfg["train"]["train_max_per_class"] = args.train_max_per_class
        # train_max_per_class is a sampling control, not a protocol-level parameter
    if args.caption_mode is not None:
        cfg["caption_mode"] = args.caption_mode
        fig_dir = Path(cfg["output_dir"])
        cfg["checkpoint_dir"] = str(fig_dir / "checkpoints" / args.caption_mode)
        cfg["eval"]["ckpt_blip"] = str(fig_dir / "checkpoints" / args.caption_mode / "blip_phase_joint_best.pth")
        if "ram" in cfg.get("senders", []):
            cfg["eval"]["ckpt_ram"] = str(fig_dir / "checkpoints" / args.caption_mode / "ram_phase_joint_best.pth")
    # ── --finetune_clip CLI 覆盖 ──
    if args.finetune_clip:
        cfg["finetune_clip"] = True
        cfg["protocol"]["required_classifier_backend"] = "clip_finetuned"
        cfg["required_classifier_backend"] = "clip_finetuned"

    if protocol_locked and protocol_override_fields and not args.allow_protocol_override:
        raise RuntimeError(
            "Fig7 protocol is locked. Refuse to override: "
            f"{sorted(set(protocol_override_fields))}. "
            "Use --allow_protocol_override only for debug runs."
        )

    # ── Formal-mode guard: refuse smoke-level training parameters ──
    # Paper reproduction requires full-dataset training. If user overrode
    # max_per_class or max_batches to tiny values, these results cannot be
    # used as formal experiments.
    _train_max_per_class = int(cfg.get("train", {}).get("train_max_per_class", -1))
    _train_batch_size = int(cfg.get("train", {}).get("train_batch_size", 16))
    if _train_max_per_class > 0 and _train_max_per_class < 100:
        if not args.allow_protocol_override:
            raise RuntimeError(
                f"Fig7 formal mode: train_max_per_class={_train_max_per_class} is too small "
                f"for paper reproduction (need -1 for full dataset or >=100). "
                f"This would produce smoke-level results. Use --allow_protocol_override to force."
            )
        else:
            print(
                f"[FIG7-WARN] train_max_per_class={_train_max_per_class} — "
                f"this is a SMOKE/DEBUG run, results must NOT be used as formal experiments."
            )

    _validate_fig7_protocol(cfg, mode=args.mode)
    validate_paper_repro_config("fig7", cfg)

    # ── smoke / formal 隔离 + 正式准入门卫 ──
    # Fig7 has custom smoke logic: redirect output_dir and checkpoint_dir
    from common.experiment_bootstrap import apply_smoke_formal_guard as _guard_fn
    _smoke = is_smoke_run(cfg)
    if _smoke:
        cfg["output_dir"] = resolve_output_dir(cfg["output_dir"], "fig7", cfg)
        cfg["checkpoint_dir"] = resolve_output_dir(cfg["checkpoint_dir"], "fig7", cfg)
        print(f"[FIG7-SMOKE] 输出重定向至: {cfg['output_dir']}")
    _guard_fn("fig7", cfg)

    inject_provenance(cfg)

    print_experiment_params("fig7", args.mode, cfg)

    snapshot_path = _snapshot_config(cfg, mode=args.mode)
    print(f"[FIG7] config snapshot: {snapshot_path}")

    # ── CLIP 分类器微调（评估前执行） ──────────────────────────────────────
    if cfg.get("finetune_clip", False):
        clip_path = cfg.get("clip_classifier_path", "")
        clip_exists = clip_path and Path(clip_path).exists()

        do_finetune = True
        if clip_exists:
            print(f"\n[FIG7] 发现已有微调 CLIP 权重: {clip_path}")
            while True:
                answer = input("  是否使用已有权重？(y=使用已有 / n=重新微调): ").strip().lower()
                if answer in ("y", "yes", "是"):
                    do_finetune = False
                    print("  → 使用已有权重，跳过 CLIP 微调。")
                    break
                elif answer in ("n", "no", "否"):
                    do_finetune = True
                    print("  → 重新微调 CLIP 分类器 ...")
                    break
                else:
                    print("  请输入 y 或 n。")

        if do_finetune:
            print("\n[FIG7] ═══ 启动 CLIP 分类器微调 ═══")
            _run_clip_finetune(cfg)
            print(f"[FIG7] CLIP 分类器已保存: {cfg['clip_classifier_path']}")

    if args.mode in {"train", "all"} and cfg["train"]["enabled"]:
        reuse_best = bool(cfg.get("train", {}).get("use_previous_best_checkpoint", False))
        eval_cfg = cfg.get("eval", {})
        ckpt_blip = Path(str(eval_cfg.get("ckpt_blip", ""))).expanduser().resolve()
        need_ram = "ram" in cfg.get("senders", [])
        ckpt_ram = Path(str(eval_cfg.get("ckpt_ram", ""))).expanduser().resolve() if need_ram else None
        has_best_ckpt = ckpt_blip.exists() and (not need_ram or ckpt_ram.exists())

        if reuse_best and has_best_ckpt:
            print(
                "[FIG7] train.use_previous_best_checkpoint=True and both best checkpoints exist; "
                "skip training and reuse existing best models."
            )
        else:
            train_cfg = _build_fig7_train_config(cfg)
            train_result = run_training(train_cfg)
            print(f"[FIG7] train summary: {train_result['summary_csv']}")

    if args.mode in {"eval", "all"} and cfg["eval"]["enabled"]:
        _resolve_fig7_eval_checkpoints(cfg)
        eval_cfg = EvalConfig(
            project_root=cfg["project_root"],
            model_file=cfg["model_file"],
            target_file=cfg["target_file"],
            test_split_dir=cfg["test_split_dir"],
            output_dir=cfg["output_dir"],
            senders=cfg["senders"],
            blip_ckb_dir=cfg["blip_ckb_dir"],
            ram_ckb_path=cfg.get("ram_ckb_path", ""),
            sd_ckb_dir=cfg["sd_ckb_dir"],
            channel_type=cfg["channel_type"],
            ckpt_blip=cfg["eval"]["ckpt_blip"],
            ckpt_ram=cfg["eval"].get("ckpt_ram", ""),
            snr_list=cfg["snr_list"],
            sd_steps=cfg["sd_steps"],
            sd_height=cfg.get("sd_height", 512),
            sd_width=cfg.get("sd_width", 512),
            batch_size=cfg["eval"]["batch_size"],
            max_batches=cfg["eval"]["max_batches"],
            max_per_class=cfg["eval"]["max_per_class"],
            seed=cfg["seed"],
            strict_ckpt=cfg["strict_ckpt"],
            strict_paper_repro=STRICT_PAPER_REPRO,
            quiet_third_party=cfg["quiet_third_party"],
            tag=cfg["eval"]["tag"],
            metrics=cfg.get("metrics", ["ssq"]),
            baselines=cfg.get("baselines"),
            max_text_len=int(cfg.get("max_text_len", 24)),
            max_text_len_by_sender=cfg.get("max_text_len_by_sender"),
            required_classifier_backend=cfg["protocol"]["required_classifier_backend"],
            clip_classifier_path=cfg.get("clip_classifier_path", ""),
            channel_dim=cfg.get("channel_dim"),
            caption_mode=cfg.get("caption_mode", "baseline"),
            caption_prompt=cfg.get("caption_prompt"),
        )
        eval_result = run_evaluation(eval_cfg)
        print(f"[FIG7] eval curve: {eval_result['curve_csv']}")
        print(f"[FIG7] eval details: {eval_result['detail_csv']}")
        print(f"[FIG7] eval plot: {eval_result['plot_png']}")


if __name__ == "__main__":
    main()
