"""Prepare BLIP text cache for datasets.

Formal mode requires real pretrained BLIP load success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import torch
from torch.utils.data import DataLoader

from data.cache import CaptionCacheItem, ensure_cache_dirs, write_caption_cache
from data.datasets import BirdsDataset, CIFARDataset, CatsVsDogsDataset
from models.kb_blip import BlipKnowledgeBase


def _build_dataloader(dataset_name: str, data_root: Path, image_size: int, batch_size: int) -> DataLoader:
    if dataset_name == "cifar":
        cifar_candidates = [
            data_root / "cifar10",
            data_root,
            data_root / "cifar-10-batches-py",
        ]
        cifar_root = None
        for candidate in cifar_candidates:
            if (candidate / "cifar-10-batches-py").exists() or (candidate / "data_batch_1").exists():
                cifar_root = candidate
                break
        if cifar_root is None:
            cifar_root = data_root / "cifar10"
        dataset = CIFARDataset(root=str(cifar_root), train=True, image_size=image_size, download=False)
    elif dataset_name == "birds":
        dataset = BirdsDataset(root=str(data_root / "birds"), split="train", image_size=image_size)
    elif dataset_name == "catsvsdogs":
        dataset = CatsVsDogsDataset(root=str(data_root / "catsvsdogs"), split="train", image_size=image_size)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cifar", "birds", "catsvsdogs"])
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--cache-root", type=str, default="outputs")
    parser.add_argument("--model-name", type=str, default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--model-cache-dir", type=str, default="D:/model_cache/vlm_csc/blip")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--max-samples-per-dataset", type=int, default=200)
    return parser.parse_args()


def run_prepare_text_cache(
    datasets: Iterable[str],
    data_root: Path,
    cache_root: Path,
    model_name: str,
    model_cache_dir: str,
    image_size: int,
    batch_size: int,
    max_length: int,
    allow_fallback: bool,
    max_samples_per_dataset: int,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[prepare_text_cache] device={device}")
    print(f"[prepare_text_cache] model={model_name}")
    print(f"[prepare_text_cache] allow_fallback={allow_fallback}")

    kb = BlipKnowledgeBase(
        checkpoint=model_name,
        max_length=max_length,
        device=device,
        load_pretrained=True,
        allow_fallback=allow_fallback,
        cache_dir=model_cache_dir,
    )

    dirs = ensure_cache_dirs(cache_root)
    caption_dir = Path(dirs["caption_jsonl_dir"])
    summary = {"datasets": {}, "caption_dir": str(caption_dir)}

    for ds_name in datasets:
        records: List[CaptionCacheItem] = []
        processed = 0
        try:
            loader = _build_dataloader(ds_name, data_root=data_root, image_size=image_size, batch_size=batch_size)
            for batch in loader:
                images = batch["image"].to(device)
                sample_ids = batch["sample_id"]
                dataset_names = batch["dataset_name"]

                out = kb.generate_caption_tokens(images)
                captions = out["captions"]
                input_ids = out["input_ids"].cpu().tolist()

                for sid, dsn, cap, ids in zip(sample_ids, dataset_names, captions, input_ids):
                    records.append(
                        CaptionCacheItem(
                            sample_id=str(sid),
                            dataset_name=str(dsn),
                            caption=str(cap),
                            tokenizer_ids=[int(v) for v in ids],
                        )
                    )
                    processed += 1
                    if processed >= max_samples_per_dataset:
                        break
                if processed >= max_samples_per_dataset:
                    break

            jsonl_path = caption_dir / f"{ds_name}_train.jsonl"
            write_caption_cache(jsonl_path, records)
            summary["datasets"][ds_name] = {
                "status": "ok",
                "samples": len(records),
                "cache_file": str(jsonl_path),
            }
        except Exception as exc:
            summary["datasets"][ds_name] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    return summary


def main() -> None:
    args = parse_args()
    summary = run_prepare_text_cache(
        datasets=args.datasets,
        data_root=Path(args.data_root),
        cache_root=Path(args.cache_root),
        model_name=args.model_name,
        model_cache_dir=args.model_cache_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_length=args.max_length,
        allow_fallback=args.allow_fallback,
        max_samples_per_dataset=args.max_samples_per_dataset,
    )
    out_path = Path(args.cache_root) / "cache" / "captions" / "prepare_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TEXT_CACHE_PREPARE_DONE")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
