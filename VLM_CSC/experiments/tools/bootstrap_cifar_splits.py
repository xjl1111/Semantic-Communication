from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def _load_cifar_batch(path: Path):
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="bytes")
    data = obj[b"data"]
    labels = obj.get(b"labels", obj.get(b"fine_labels"))
    if labels is None:
        raise RuntimeError(f"No labels found in CIFAR batch: {path}")
    return data, labels


def _save_samples(data: np.ndarray, labels: list[int], out_root: Path, prefix: str):
    out_root.mkdir(parents=True, exist_ok=True)
    for class_id in range(10):
        (out_root / str(class_id)).mkdir(parents=True, exist_ok=True)

    for idx, (flat, label) in enumerate(zip(data, labels)):
        arr = np.asarray(flat, dtype=np.uint8).reshape(3, 32, 32).transpose(1, 2, 0)
        image = Image.fromarray(arr, mode="RGB")
        image.save(out_root / str(int(label)) / f"{prefix}_{idx:05d}.png")


def main():
    project_root = Path(__file__).resolve().parents[3]
    src_root = project_root / "data" / "datasets" / "cifar-10-batches-py"
    dst_root = project_root / "data" / "datasets" / "cifar"
    train_root = dst_root / "train"
    test_root = dst_root / "test"

    if not src_root.exists():
        raise RuntimeError(f"CIFAR source dir not found: {src_root}")

    if train_root.exists() and test_root.exists():
        train_has_png = any(train_root.rglob("*.png"))
        test_has_png = any(test_root.rglob("*.png"))
        if train_has_png and test_has_png:
            print(f"[SKIP] existing CIFAR split images detected at: {dst_root}")
            return

    train_chunks = []
    train_labels = []
    for i in range(1, 6):
        data, labels = _load_cifar_batch(src_root / f"data_batch_{i}")
        train_chunks.append(np.asarray(data))
        train_labels.extend(int(v) for v in labels)

    train_data = np.concatenate(train_chunks, axis=0)
    _save_samples(train_data, train_labels, train_root, prefix="train")

    test_data, test_labels = _load_cifar_batch(src_root / "test_batch")
    _save_samples(np.asarray(test_data), [int(v) for v in test_labels], test_root, prefix="test")

    print(f"[DONE] CIFAR split images generated at: {dst_root}")


if __name__ == "__main__":
    main()
