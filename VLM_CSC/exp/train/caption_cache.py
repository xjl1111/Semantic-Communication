"""Caption cache management — extracted from train_experiment.py for modularity.

Provides strict-mode caption caching with dataset-hash verification,
legacy path remapping, RAM format consistency guard, and auto-rebuild.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

from PIL import Image
from tqdm import tqdm


# ─────────────────────────── helpers ───────────────────────────


def dataset_hash(records: List[Dict]) -> str:
    """SHA-256 fingerprint of sorted image paths in *records*."""
    payload = "\n".join(sorted(str(rec["path"]) for rec in records))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _map_legacy_path(path_str: str) -> str:
    """Map ``\\data\\datasets\\`` → ``\\data\\`` for backward-compat."""
    p = str(path_str)
    if "\\data\\datasets\\" in p:
        return p.replace("\\data\\datasets\\", "\\data\\")
    return p


def load_any_entries(cache_file: Path) -> Dict[str, str]:
    """Best-effort load of any cache file (v1 flat *or* v2 with ``_meta``)."""
    if not cache_file.exists():
        return {}
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}

    if isinstance(payload.get("captions"), dict):
        source = payload["captions"]
    else:
        source = {k: v for k, v in payload.items() if not str(k).startswith("_")}

    return {str(k).strip(): str(v).strip() for k, v in source.items() if str(k).strip() and str(v).strip()}


# ─────────────────── strict load / save ────────────────────────


def load_strict(
    records: List[Dict],
    sender: str,
    cache_file: Path,
    strict_required: bool,
    caption_prompt: str | None = None,
    caption_mode: str | None = None,
) -> Dict[str, str]:
    """Load caption cache with full integrity checks.

    除了检查 dataset_hash 和 sender 外，还会检查 caption_prompt 和 caption_mode。
    这样只有真正影响 caption 生成的参数（prompt、mode、数据集路径）变化才会
    触发重建，而 channel_dim、finetune_clip 等无关参数不会。

    Raises ``RuntimeError`` (with descriptive message) on every
    integrity violation so that the caller can decide to auto-rebuild.
    """
    if not cache_file.exists():
        if strict_required:
            raise RuntimeError(
                f"Caption cache missing under strict mode for sender={sender}: {cache_file}. "
                "Please prebuild cache explicitly before training."
            )
        return {}

    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid caption cache format: {cache_file}")

    if "_meta" not in payload or "captions" not in payload:
        raise RuntimeError(
            f"Caption cache schema invalid under strict mode (missing _meta/captions): {cache_file}. "
            "Please migrate cache explicitly before training."
        )

    meta, captions = payload["_meta"], payload["captions"]
    if not isinstance(meta, dict) or not isinstance(captions, dict):
        raise RuntimeError(f"Caption cache malformed _meta/captions: {cache_file}")

    if meta.get("sender") != sender:
        raise RuntimeError(
            f"Caption cache sender mismatch: expected={sender}, got={meta.get('sender')}, file={cache_file}"
        )

    expected = dataset_hash(records)
    if meta.get("dataset_hash") != expected:
        raise RuntimeError(
            f"Caption cache dataset hash mismatch: expected={expected}, "
            f"got={meta.get('dataset_hash')}, file={cache_file}"
        )

    # ── 检查 caption_prompt / caption_mode / sr_enabled ──
    # 这三个参数直接影响 caption 生成结果：
    #   - caption_prompt: BLIP 的文本前缀
    #   - caption_mode: 决定预处理流程（是否 SR、用哪个模型）
    #   - sr_enabled: SR 会将 32×32 上采样至 256×256，显著改变 BLIP 输入
    cached_prompt = meta.get("caption_prompt")
    cached_mode = meta.get("caption_mode")
    cached_sr = meta.get("sr_enabled")
    # 向后兼容：旧 cache 没有这些字段时跳过检查
    if cached_prompt is not None and caption_prompt is not None:
        if cached_prompt != caption_prompt:
            raise RuntimeError(
                f"Caption cache prompt mismatch: cached='{cached_prompt}', "
                f"current='{caption_prompt}'. Prompt 变化会影响 caption 生成结果，需要重建缓存。"
            )
    if cached_mode is not None and caption_mode is not None:
        if cached_mode != caption_mode:
            raise RuntimeError(
                f"Caption cache mode mismatch: cached='{cached_mode}', "
                f"current='{caption_mode}'. Mode 变化会影响 caption 生成结果，需要重建缓存。"
            )
    # SR 开关检查：即使 mode 名称相同，也要确认 SR 状态一致
    current_sr = _is_sr_mode(caption_mode)
    if cached_sr is not None and caption_mode is not None:
        if cached_sr != current_sr:
            raise RuntimeError(
                f"Caption cache SR mismatch: cached sr_enabled={cached_sr}, "
                f"current sr_enabled={current_sr} (mode='{caption_mode}'). "
                f"SR 上采样会显著改变 BLIP 输入图像，需要重建缓存。"
            )

    missing = [str(r["path"]) for r in records if str(r["path"]) not in captions]
    if missing:
        raise RuntimeError(f"Caption cache incomplete for sender={sender}, missing entries={len(missing)}")

    return captions


def save_strict(
    *,
    cache_file: Path,
    sender: str,
    records: List[Dict],
    captions: Dict[str, str],
    caption_prompt: str | None = None,
    caption_mode: str | None = None,
) -> None:
    """Persist caption cache with v2 schema (``_meta`` + ``captions``)."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    required_keys = [str(r["path"]) for r in records]
    missing = [k for k in required_keys if k not in captions or str(captions[k]).strip() == ""]
    if missing:
        raise RuntimeError(f"Cannot save strict cache: missing captions for {len(missing)} records")

    meta: Dict = {
        "sender": str(sender),
        "dataset_hash": dataset_hash(records),
        "num_records": len(records),
        "cache_format_version": 2,
    }
    if caption_prompt is not None:
        meta["caption_prompt"] = caption_prompt
    if caption_mode is not None:
        meta["caption_mode"] = caption_mode
        meta["sr_enabled"] = _is_sr_mode(caption_mode)

    payload = {
        "_meta": meta,
        "captions": {k: str(captions[k]).strip() for k in required_keys},
    }
    cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ─────────────── SR mode helper ────────────────────────────────

# 需要启用 SR 的 caption 模式（与 VLM-CSC.py 中 _SR_ENABLED_MODES 保持一致）
_SR_MODES = frozenset({"sr", "sr_prompt", "blip2", "baseline"})


def _is_sr_mode(mode: str | None) -> bool:
    """判断给定 caption_mode 是否启用了 SR 上采样。"""
    return str(mode).strip().lower() in _SR_MODES if mode else False


# ─────────────── build / rebuild / RAM guard ───────────────────


def build_missing(
    *,
    model,
    sender: str,
    records: List[Dict],
    caption_cache: Dict[str, str],
    save_file: Path | None = None,
    save_interval: int = 500,
    caption_prompt: str | None = None,
    caption_mode: str | None = None,
) -> Dict[str, str]:
    """Generate captions for *records* not yet in *caption_cache*.

    支持增量保存：每生成 *save_interval* 条 caption 就写入一次磁盘（v2 格式），
    这样即使中途中断，已生成的部分不会丢失，下次运行只需继续生成剩余部分。
    """
    missing_recs = [r for r in records if not caption_cache.get(str(r["path"]), "").strip()]
    if not missing_recs:
        return caption_cache

    print(f"[CAPTION_CACHE] sender={sender} missing={len(missing_recs)}; generating captions on-the-fly")
    generated_since_save = 0
    for rec in tqdm(missing_recs, desc=f"caption bootstrap ({sender})", leave=False):
        text = str(model.sender_ckb.forward(Image.open(rec["path"]).convert("RGB"))).strip()
        if not text:
            raise RuntimeError(f"Generated empty caption for image={rec['path']}")
        caption_cache[str(rec["path"])] = text
        generated_since_save += 1

        # 增量保存：防止中断丢失进度
        if save_file is not None and generated_since_save >= save_interval:
            _incremental_save(save_file, sender, records, caption_cache,
                              caption_prompt=caption_prompt, caption_mode=caption_mode)
            generated_since_save = 0

    # 结束后最终保存一次
    if save_file is not None and generated_since_save > 0:
        _incremental_save(save_file, sender, records, caption_cache,
                          caption_prompt=caption_prompt, caption_mode=caption_mode)

    return caption_cache


def _incremental_save(
    cache_file: Path,
    sender: str,
    records: List[Dict],
    captions: Dict[str, str],
    *,
    caption_prompt: str | None = None,
    caption_mode: str | None = None,
) -> None:
    """增量保存当前已有的 captions（允许不完整，用于断点续传）。

    同样将 caption_prompt / caption_mode / sr_enabled 写入 _meta，
    确保中断重启后仍能正确校验缓存有效性。
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    done = sum(1 for r in records if captions.get(str(r["path"]), "").strip())
    meta: Dict = {
        "sender": str(sender),
        "dataset_hash": dataset_hash(records),
        "num_records": len(records),
        "num_cached": done,
        "cache_format_version": 2,
        "partial": done < len(records),
    }
    if caption_prompt is not None:
        meta["caption_prompt"] = caption_prompt
    if caption_mode is not None:
        meta["caption_mode"] = caption_mode
        meta["sr_enabled"] = _is_sr_mode(caption_mode)
    payload = {
        "_meta": meta,
        "captions": {str(k): str(v).strip() for k, v in captions.items() if str(v).strip()},
    }
    cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\r[CAPTION_CACHE] 增量保存: {done}/{len(records)} 条 → {cache_file.name}", end="")


def rebuild_for_records(
    *,
    model,
    sender: str,
    records: List[Dict],
    cache_file: Path,
    caption_prompt: str | None = None,
    caption_mode: str | None = None,
) -> Dict[str, str]:
    """Rebuild cache from scratch, reusing existing entries + legacy remap.

    支持断点续传：如果 cache_file 已有部分条目（上次中断遗留），
    会自动跳过已完成的条目，只生成缺失的部分。
    """
    existing = load_any_entries(cache_file)

    # 同时尝试从兄弟模式缓存中借用已有 caption（仅限同 SR 类别）
    # SR 会改变输入图片（32×32→256×256），所以 SR 模式和非 SR 模式的 caption 不可互借
    _sibling_modes = ["prompt", "sr_prompt", "baseline", "sr", "blip2"]
    current_sr = _is_sr_mode(caption_mode)
    cache_dir = cache_file.parent.parent  # caption_cache/
    for sib_mode in _sibling_modes:
        # 只从同 SR 类别的兄弟借用
        if _is_sr_mode(sib_mode) != current_sr:
            continue
        sib_file = cache_dir / sib_mode / cache_file.name
        if sib_file.exists() and sib_file != cache_file:
            sib_entries = load_any_entries(sib_file)
            if sib_entries:
                # 只借用当前缺失的条目
                borrowed = 0
                for rec in records:
                    key = str(rec["path"])
                    if not existing.get(key, "").strip() and sib_entries.get(key, "").strip():
                        existing[key] = sib_entries[key]
                        borrowed += 1
                if borrowed > 0:
                    print(f"[CAPTION_CACHE] 从 {sib_mode}/ 借用了 {borrowed} 条已有 caption (同SR类别={current_sr})")

    remapped: Dict[str, str] = {}
    for rec in records:
        key = str(rec["path"])
        val = existing.get(key, "").strip()
        if val:
            remapped[key] = val
            continue
        legacy = _map_legacy_path(key)
        lval = existing.get(legacy, "").strip()
        if lval:
            remapped[key] = lval

    rebuilt = build_missing(
        model=model, sender=sender, records=records, caption_cache=remapped,
        save_file=cache_file, save_interval=500,
        caption_prompt=caption_prompt, caption_mode=caption_mode,
    )
    save_strict(cache_file=cache_file, sender=sender, records=records, captions=rebuilt,
                 caption_prompt=caption_prompt, caption_mode=caption_mode)
    return rebuilt


def _check_ram_format(captions: Dict[str, str]) -> bool:
    """Return True if any sampled RAM captions have wrong format."""
    if not captions:
        return False
    sample = list(captions.values())[: min(20, len(captions))]
    return any(not s.startswith("This image contains:") for s in sample)


def load_with_optional_rebuild(
    *,
    model,
    sender: str,
    records: List[Dict],
    cache_file: Path,
    strict_required: bool,
    auto_rebuild_on_hash_mismatch: bool,
    caption_prompt: str | None = None,
    caption_mode: str | None = None,
) -> Dict[str, str]:
    """High-level loader: try strict load → auto-rebuild on known errors → RAM guard."""
    try:
        captions = load_strict(
            records=records, sender=sender, cache_file=cache_file,
            strict_required=strict_required,
            caption_prompt=caption_prompt, caption_mode=caption_mode,
        )
    except RuntimeError as exc:
        msg = str(exc).lower()
        if auto_rebuild_on_hash_mismatch and (
            "caption cache missing" in msg
            or "dataset hash mismatch" in msg
            or "caption cache schema invalid" in msg
            or "caption cache incomplete" in msg
            or "caption cache sender mismatch" in msg
            or "caption cache malformed" in msg
            or "prompt mismatch" in msg
            or "mode mismatch" in msg
            or "sr mismatch" in msg
        ):
            print(f"[CAPTION_CACHE] sender={sender} cache不可用 ({cache_file.name}); 自动重建...")
            return rebuild_for_records(
                model=model, sender=sender, records=records, cache_file=cache_file,
                caption_prompt=caption_prompt, caption_mode=caption_mode,
            )
        raise

    # RAM format consistency guard
    if str(sender).lower() == "ram" and _check_ram_format(captions):
        sample_n = min(20, len(captions))
        wrong = sum(1 for s in list(captions.values())[:sample_n] if not s.startswith("This image contains:"))
        print(
            f"[CAPTION_CACHE] sender=ram: detected {wrong}/{sample_n} captions "
            f"with wrong format (expected 'This image contains: ...'). "
            f"Forcing full cache rebuild to ensure format consistency."
        )
        rebuilt = build_missing(model=model, sender=sender, records=records, caption_cache={},
                               caption_prompt=caption_prompt, caption_mode=caption_mode)
        save_strict(cache_file=cache_file, sender=sender, records=records, captions=rebuilt,
                     caption_prompt=caption_prompt, caption_mode=caption_mode)
        return rebuilt

    return captions


# ──────────── convenience: load-or-build for a sender ──────────


def ensure_captions_for_sender(
    *,
    model,
    sender: str,
    records: List[Dict],
    cache_file: Path,
    use_caption_cache: bool,
    strict_cache_required: bool,
    auto_rebuild: bool,
    caption_prompt: str | None = None,
    caption_mode: str | None = None,
) -> Dict[str, str]:
    """One-call helper used by both ``train_sender`` and fig8 continual loop."""
    if not use_caption_cache:
        return build_missing(model=model, sender=sender, records=records, caption_cache={},
                             caption_prompt=caption_prompt, caption_mode=caption_mode)

    captions = load_with_optional_rebuild(
        model=model,
        sender=sender,
        records=records,
        cache_file=cache_file,
        strict_required=strict_cache_required,
        auto_rebuild_on_hash_mismatch=auto_rebuild,
        caption_prompt=caption_prompt,
        caption_mode=caption_mode,
    )
    if not strict_cache_required:
        captions = build_missing(model=model, sender=sender, records=records, caption_cache=captions,
                                 caption_prompt=caption_prompt, caption_mode=caption_mode)
        save_strict(cache_file=cache_file, sender=sender, records=records, captions=captions,
                     caption_prompt=caption_prompt, caption_mode=caption_mode)
    return captions
