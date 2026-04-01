"""通用工具函数与常量。"""
from __future__ import annotations

import importlib.util
import logging
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List


LABEL_MAP = {"cat": 0, "dog": 1}


def configure_runtime_logging(quiet_third_party: bool) -> None:
    if not quiet_third_party:
        return

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"fairscale\.experimental\.nn\.offload",
    )

    root_logger = logging.getLogger()
    if root_logger.level < logging.ERROR:
        root_logger.setLevel(logging.ERROR)

    for logger_name in ["transformers", "diffusers", "fairscale", "accelerate"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def collect_binary_images_from_split(split_dir: Path, max_per_class: int) -> List[Dict]:
    records = []
    for cls_name in ["cat", "dog"]:
        class_dir = split_dir / cls_name
        if not class_dir.exists():
            continue
        image_paths = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if max_per_class >= 0:
            image_paths = image_paths[:max_per_class]
        for image_path in image_paths:
            records.append({"path": image_path, "label": LABEL_MAP[cls_name]})
    return records


def collect_generic_images_from_split(split_dir: Path, max_per_class: int) -> List[Dict]:
    records = []
    class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    for class_idx, class_dir in enumerate(sorted(class_dirs)):
        image_paths = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if max_per_class >= 0:
            image_paths = image_paths[:max_per_class]
        for image_path in image_paths:
            records.append({"path": image_path, "label": class_idx})
    return records


def chunk_records(records: List[Dict], batch_size: int, max_batches: int = -1) -> List[List[Dict]]:
    if batch_size <= 0:
        raise RuntimeError(f"batch_size must be positive, got {batch_size}")
    batches = [records[i : i + batch_size] for i in range(0, len(records), batch_size)]
    if max_batches is not None and max_batches > 0:
        batches = batches[:max_batches]
    return batches
