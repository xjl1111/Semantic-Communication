"""Step 2 data pipeline tests.

These tests validate:
- deterministic continual split generation and persistence
- caption cache jsonl schema roundtrip
"""

from pathlib import Path

from data.cache import CaptionCacheItem, append_caption_cache, ensure_cache_dirs, read_caption_cache
from data.continual_splits import build_task_splits, load_splits_json, save_splits_json, save_task_sizes


def test_continual_split_save_load(tmp_path: Path) -> None:
    meta_dir = tmp_path / "meta"
    save_task_sizes({"cifar": 100, "birds": 50, "catsvsdogs": 40}, output_dir=meta_dir)
    splits = build_task_splits(seed=123, output_dir=meta_dir)
    out = tmp_path / "splits.json"
    save_splits_json(splits, out)
    loaded = load_splits_json(out)
    assert loaded == splits
    assert len(loaded["cifar"]["train"]) == 80
    assert len(loaded["birds"]["val"]) == 5


def test_caption_cache_roundtrip(tmp_path: Path) -> None:
    dirs = ensure_cache_dirs(tmp_path)
    jsonl_path = dirs["caption_jsonl_dir"] / "cifar_train.jsonl"

    append_caption_cache(
        jsonl_path,
        CaptionCacheItem(
            sample_id="cifar:train:0",
            dataset_name="cifar",
            caption="a small object",
            tokenizer_ids=[101, 102, 103],
        ),
    )

    records = read_caption_cache(jsonl_path)
    assert len(records) == 1
    assert records[0].dataset_name == "cifar"
    assert isinstance(records[0].tokenizer_ids, list)
