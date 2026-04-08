from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from PIL import Image
from tqdm import tqdm

import sys

_THIS_DIR = Path(__file__).resolve().parent
_EXP_DIR = _THIS_DIR.parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from common import (
    TaskDatasetManager,
    build_vlm_system,
    collect_binary_images_from_split,
)
from fig7.fig7_config import build_fig7_config
from fig8.fig8_config import build_fig8_config
from fig9.fig9_config import build_fig9_config
from fig10.fig10_config import build_fig10_config


def _dataset_hash(records: List[Dict]) -> str:
    import hashlib

    payload = "\n".join(sorted(str(Path(rec["path"]).expanduser().resolve()) for rec in records))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_any_cache(path: Path) -> Dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    if "captions" in payload and isinstance(payload["captions"], dict):
        return {str(k): str(v) for k, v in payload["captions"].items() if str(v).strip() != ""}
    return {str(k): str(v) for k, v in payload.items() if not str(k).startswith("_") and str(v).strip() != ""}


def _old_path_candidate(new_path: Path) -> Path:
    s = str(new_path)
    s = s.replace("\\data\\datasets\\", "\\data\\")
    return Path(s)


def _save_strict_cache(cache_file: Path, sender: str, records: List[Dict], captions: Dict[str, str]) -> None:
    keys = [str(Path(rec["path"]).expanduser().resolve()) for rec in records]
    missing = [k for k in keys if k not in captions or str(captions[k]).strip() == ""]
    if missing:
        raise RuntimeError(f"Cannot save strict cache; missing captions={len(missing)} file={cache_file}")

    payload = {
        "_meta": {
            "sender": str(sender),
            "dataset_hash": _dataset_hash(records),
            "num_records": len(records),
        },
        "captions": {k: str(captions[k]).strip() for k in keys},
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _collect_sender_records_fig7(cfg: dict, max_per_class_override: int | None) -> List[Dict]:
    split_dir = Path(cfg["train_split_dir"])
    max_per_class = int(cfg["train"]["train_max_per_class"]) if max_per_class_override is None else int(max_per_class_override)
    return collect_binary_images_from_split(split_dir, max_per_class)


def _collect_sender_records_fig9(cfg: dict, max_per_class_override: int | None) -> List[Dict]:
    split_dir = Path(cfg["train_split_dir"])
    max_per_class = int(cfg["train"]["train_max_per_class"]) if max_per_class_override is None else int(max_per_class_override)
    return collect_binary_images_from_split(split_dir, max_per_class)


def _collect_sender_records_fig10(cfg: dict, max_per_class_override: int | None) -> List[Dict]:
    split_dir = Path(cfg["train_split_dir"])
    max_per_class = int(cfg["train"]["train_max_per_class"]) if max_per_class_override is None else int(max_per_class_override)
    return collect_binary_images_from_split(split_dir, max_per_class)


def _collect_sender_records_fig8(cfg: dict, task_name: str, max_per_class_override: int | None) -> List[Dict]:
    max_per_class = int(cfg["train"]["train_max_per_class"]) if max_per_class_override is None else int(max_per_class_override)
    manager = TaskDatasetManager(
        sequence=cfg["dataset_sequence"],
        dataset_roots=cfg["dataset_roots"],
        dataset_splits=cfg["dataset_splits"],
        max_per_class=max_per_class,
        val_split_ratio=float(cfg.get("val_split_ratio", 0.2)),
        val_split_seed=int(cfg.get("val_split_seed", 42)),
        strict_mode=True,
        consumer="audit",
    )
    return manager.get_task_train_set(task_name) + manager.get_task_val_set(task_name)


def _build_sender_model(cfg: dict, model_file: str, sender: str):
    return build_vlm_system(
        sender=sender,
        blip_dir=Path(cfg["blip_ckb_dir"]),
        ram_ckpt=Path(cfg["ram_ckb_path"]),
        sd_dir=Path(cfg["sd_ckb_dir"]),
        channel_type=cfg["channel_type"],
        device="cuda",
        quiet_third_party=bool(cfg.get("quiet_third_party", True)),
        use_real_receiver_ckb=False,
        enable_med=False,
        med_kwargs=None,
        max_text_len=int(cfg.get("max_text_len", 24)),
        max_text_len_by_sender=cfg.get("max_text_len_by_sender"),
    )


def _fill_captions(model, sender: str, records: List[Dict], existing: Dict[str, str], generate_missing: bool) -> Dict[str, str]:
    captions = dict(existing)
    missing = []
    for rec in records:
        p = Path(rec["path"]).expanduser().resolve()
        key = str(p)
        if key in captions and str(captions[key]).strip() != "":
            continue
        old_key = str(_old_path_candidate(p))
        if old_key in captions and str(captions[old_key]).strip() != "":
            captions[key] = str(captions[old_key]).strip()
            continue
        missing.append(p)

    if missing and not generate_missing:
        raise RuntimeError(
            f"Strict cache rebuild found missing captions={len(missing)} for sender={sender}. "
            "Re-run with --generate-missing to explicitly generate missing entries."
        )

    for image_path in tqdm(missing, desc=f"caption generate ({sender})", leave=False):
        image = Image.open(image_path).convert("RGB")
        text = str(model.sender_ckb.forward(image)).strip()
        if text == "":
            raise RuntimeError(f"Generated empty caption: {image_path}")
        captions[str(image_path)] = text

    return captions


def _rebuild_single_cache(cache_file: Path, model, sender: str, records: List[Dict], generate_missing: bool) -> None:
    existing = _load_any_cache(cache_file)
    captions = _fill_captions(model, sender, records, existing, generate_missing)
    _save_strict_cache(cache_file, sender, records, captions)
    print(f"[CACHE] sender={sender} records={len(records)} saved={cache_file}")


def rebuild_fig7(generate_missing: bool, max_per_class_override: int | None) -> None:
    cfg = build_fig7_config()
    records = _collect_sender_records_fig7(cfg, max_per_class_override)
    for sender in cfg["senders"]:
        model = _build_sender_model(cfg, cfg["model_file"], sender)
        cache_file = Path(cfg["caption_cache_dir"]) / f"{sender}_captions.json"
        _rebuild_single_cache(cache_file, model, sender, records, generate_missing)


def rebuild_fig8(generate_missing: bool, max_per_class_override: int | None) -> None:
    cfg = build_fig8_config()
    for variant in ["with_med", "without_med"]:
        for sender in cfg["senders"]:
            model = _build_sender_model(cfg, cfg["model_file"], sender)
            for task in cfg["dataset_sequence"]:
                records = _collect_sender_records_fig8(cfg, task, max_per_class_override)
                cache_file = Path(cfg["caption_cache_dir"]) / variant / sender / f"task_{cfg['dataset_sequence'].index(task)+1}_{task}" / f"{sender}_captions.json"
                _rebuild_single_cache(cache_file, model, sender, records, generate_missing)


def rebuild_fig9(generate_missing: bool, max_per_class_override: int | None) -> None:
    cfg = build_fig9_config()
    records = _collect_sender_records_fig9(cfg, max_per_class_override)
    for exp_name in cfg["nam_experiments"].keys():
        for sender in cfg["senders"]:
            model = _build_sender_model(cfg, cfg["model_file"], sender)
            cache_file = Path(cfg["caption_cache_dir"]) / exp_name / f"{sender}_captions.json"
            _rebuild_single_cache(cache_file, model, sender, records, generate_missing)


def rebuild_fig10(generate_missing: bool, max_per_class_override: int | None) -> None:
    cfg = build_fig10_config()
    records = _collect_sender_records_fig10(cfg, max_per_class_override)
    for sender in cfg["senders"]:
        model = _build_sender_model(cfg, cfg["model_file"], sender)
        cache_file = Path(cfg["caption_cache_dir"]) / "vlm_csc" / f"{sender}_captions.json"
        _rebuild_single_cache(cache_file, model, sender, records, generate_missing)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicit strict caption-cache rebuild tool (no runtime fallback)")
    parser.add_argument("--fig", choices=["fig7", "fig8", "fig9", "fig10", "all"], default="all")
    parser.add_argument("--generate-missing", action="store_true")
    parser.add_argument("--max-per-class", type=int, default=None)
    args = parser.parse_args()

    selected = [args.fig] if args.fig != "all" else ["fig7", "fig8", "fig9", "fig10"]
    for fig in selected:
        print(f"[CACHE] rebuild start: {fig}")
        if fig == "fig7":
            rebuild_fig7(args.generate_missing, args.max_per_class)
        elif fig == "fig8":
            rebuild_fig8(args.generate_missing, args.max_per_class)
        elif fig == "fig9":
            rebuild_fig9(args.generate_missing, args.max_per_class)
        elif fig == "fig10":
            rebuild_fig10(args.generate_missing, args.max_per_class)
        else:
            raise RuntimeError(f"Unsupported fig: {fig}")
        print(f"[CACHE] rebuild done: {fig}")


if __name__ == "__main__":
    main()
