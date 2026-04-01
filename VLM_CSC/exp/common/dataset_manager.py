"""数据集管理器 — 任务序列的 train/val/test 分割管理。"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common.utils import collect_binary_images_from_split, collect_generic_images_from_split


class TaskDatasetManager:
    def __init__(
        self,
        sequence: List[str],
        dataset_roots: Dict[str, str] | None,
        max_per_class: int,
        *,
        dataset_splits: Dict[str, Dict[str, str]] | None = None,
        val_split_ratio: float = 0.2,
        val_split_seed: int = 42,
        strict_mode: bool = True,
        consumer: str = "audit",
    ):
        self.sequence = sequence
        self.dataset_roots = dataset_roots or {}
        self.dataset_splits = dataset_splits or {}
        self.max_per_class = max_per_class
        self.val_split_ratio = float(val_split_ratio)
        self.val_split_seed = int(val_split_seed)
        self.strict_mode = bool(strict_mode)
        self.consumer = str(consumer).lower()
        if self.consumer not in {"train", "eval", "audit"}:
            raise RuntimeError(f"TaskDatasetManager consumer must be one of ['train','eval','audit'], got {self.consumer}")

        self._task_cache: Dict[str, Dict[str, object]] = {}

    def get_task_train_set(self, task_name: str) -> List[Dict]:
        return list(self._get_task_bundle(task_name)["train_records"])

    def get_task_val_set(self, task_name: str) -> List[Dict]:
        if self.consumer == "eval":
            raise RuntimeError(f"TaskDatasetManager strict violation: evaluation phase must not read val split (task={task_name})")
        return list(self._get_task_bundle(task_name)["val_records"])

    def get_task_test_set(self, task_name: str) -> List[Dict]:
        if self.consumer == "train":
            raise RuntimeError(f"TaskDatasetManager strict violation: training phase must not read test split (task={task_name})")
        return list(self._get_task_bundle(task_name)["test_records"])

    def get_task_split_summary(self, task_name: str) -> Dict[str, object]:
        bundle = self._get_task_bundle(task_name)
        return {
            "task": task_name,
            "train_source": str(bundle["train_source"]),
            "val_source": str(bundle["val_source"]),
            "test_source": str(bundle["test_source"]),
            "num_train": len(bundle["train_records"]),
            "num_val": len(bundle["val_records"]),
            "num_test": len(bundle["test_records"]),
            "train_paths": set(bundle["train_paths"]),
            "val_paths": set(bundle["val_paths"]),
            "test_paths": set(bundle["test_paths"]),
        }

    def get_seen_tasks(self, current_task_index: int) -> List[str]:
        return self.sequence[: current_task_index + 1]

    def _resolve_task_split_paths(self, task_name: str) -> Tuple[Path, Optional[Path], Path]:
        if task_name in self.dataset_splits:
            split_cfg = self.dataset_splits[task_name]
            train_root = split_cfg["train"] if "train" in split_cfg else split_cfg.get("train_root")
            val_root = split_cfg["val"] if "val" in split_cfg else split_cfg.get("val_root")
            test_root = split_cfg["test"] if "test" in split_cfg else split_cfg.get("test_root")
            if not train_root:
                raise RuntimeError(f"TaskDatasetManager requires train split path for task={task_name}")
            if not test_root:
                raise RuntimeError(f"TaskDatasetManager requires test split path for task={task_name}")
            train_path = Path(str(train_root))
            val_path = None if val_root is None or str(val_root).strip() == "" else Path(str(val_root))
            test_path = Path(str(test_root))
            return train_path, val_path, test_path

        root = self.dataset_roots.get(task_name)
        if not root:
            raise RuntimeError(f"TaskDatasetManager missing dataset source for task={task_name}")
        base = Path(root)
        return base / "train", None, base / "test"

    def _load_records_from_dir(self, task_name: str, split_dir: Path, split_name: str) -> List[Dict]:
        if not split_dir.exists():
            raise RuntimeError(
                f"TaskDatasetManager split not found for task={task_name}, split={split_name}: {split_dir}"
            )

        if task_name == "catsvsdogs":
            records = collect_binary_images_from_split(split_dir, max_per_class=self.max_per_class)
        elif task_name == "birds":
            class_dir = split_dir / "bird"
            if not class_dir.exists():
                raise RuntimeError(f"TaskDatasetManager birds split requires 'bird' dir: {class_dir}")
            image_paths = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
            if self.max_per_class >= 0:
                image_paths = image_paths[: self.max_per_class]
            records = [{"path": image_path, "label": 0} for image_path in image_paths]
        elif task_name == "cifar":
            records = collect_generic_images_from_split(split_dir, max_per_class=self.max_per_class)
        else:
            raise RuntimeError(f"TaskDatasetManager unsupported task: {task_name}")

        if len(records) == 0:
            raise RuntimeError(
                f"TaskDatasetManager empty records for task={task_name}, split={split_name}, dir={split_dir}"
            )
        return records

    @staticmethod
    def _records_to_path_set(records: List[Dict]) -> set[str]:
        return {str(Path(rec["path"]).expanduser().resolve()) for rec in records}

    def _build_task_bundle(self, task_name: str) -> Dict[str, object]:
        train_dir, val_dir, test_dir = self._resolve_task_split_paths(task_name)

        train_full = self._load_records_from_dir(task_name, train_dir, "train")
        test_records = self._load_records_from_dir(task_name, test_dir, "test")

        if val_dir is not None:
            val_records = self._load_records_from_dir(task_name, val_dir, "val")
            train_records = train_full
            val_source = val_dir
        else:
            if not (0.0 < self.val_split_ratio < 1.0):
                raise RuntimeError(f"TaskDatasetManager val_split_ratio must be in (0,1), got {self.val_split_ratio}")
            shuffled = list(train_full)
            seed = self.val_split_seed + self.sequence.index(task_name)
            rng = random.Random(seed)
            rng.shuffle(shuffled)
            val_size = max(1, int(len(shuffled) * self.val_split_ratio))
            if val_size >= len(shuffled):
                raise RuntimeError(
                    f"TaskDatasetManager cannot split val from train for task={task_name}: train size too small ({len(shuffled)})"
                )
            val_records = shuffled[:val_size]
            train_records = shuffled[val_size:]
            val_source = f"split_from_train:{train_dir} ratio={self.val_split_ratio} seed={seed}"

        if len(val_records) == 0:
            raise RuntimeError(f"TaskDatasetManager val split is empty for task={task_name}")
        if len(test_records) == 0:
            raise RuntimeError(f"TaskDatasetManager test split is empty for task={task_name}")
        if len(train_records) == 0:
            raise RuntimeError(f"TaskDatasetManager train split is empty for task={task_name}")

        train_paths = self._records_to_path_set(train_records)
        val_paths = self._records_to_path_set(val_records)
        test_paths = self._records_to_path_set(test_records)

        if not train_paths.isdisjoint(val_paths):
            raise RuntimeError(f"TaskDatasetManager split leakage: train/val overlap for task={task_name}")
        if not train_paths.isdisjoint(test_paths):
            raise RuntimeError(f"TaskDatasetManager split leakage: train/test overlap for task={task_name}")
        if not val_paths.isdisjoint(test_paths):
            raise RuntimeError(f"TaskDatasetManager split leakage: val/test overlap for task={task_name}")

        return {
            "task": task_name,
            "train_source": train_dir,
            "val_source": val_source,
            "test_source": test_dir,
            "train_records": train_records,
            "val_records": val_records,
            "test_records": test_records,
            "train_paths": train_paths,
            "val_paths": val_paths,
            "test_paths": test_paths,
        }

    def _get_task_bundle(self, task_name: str) -> Dict[str, object]:
        if task_name not in self._task_cache:
            self._task_cache[task_name] = self._build_task_bundle(task_name)
        return self._task_cache[task_name]
