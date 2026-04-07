from __future__ import annotations

from pathlib import Path


def _build_vocab_from_tokenizer_json(blip_dir: Path) -> Path:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(blip_dir), use_fast=True, local_files_only=True)
    vocab = tokenizer.get_vocab()
    inv = [""] * len(vocab)
    for token, idx in vocab.items():
        if 0 <= int(idx) < len(inv):
            inv[int(idx)] = str(token)

    if any(v == "" for v in inv):
        raise RuntimeError("Cannot build strict vocab.txt: tokenizer vocabulary indices are not contiguous.")

    vocab_path = blip_dir / "vocab.txt"
    vocab_path.write_text("\n".join(inv), encoding="utf-8")
    return vocab_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    blip_dir = project_root / "VLM_CSC" / "data" / "assets" / "downloaded_models" / "blip"
    if not blip_dir.exists():
        raise RuntimeError(f"BLIP directory not found: {blip_dir}")

    required_base = [
        blip_dir / "config.json",
        blip_dir / "model.safetensors",
        blip_dir / "preprocessor_config.json",
        blip_dir / "tokenizer_config.json",
    ]
    missing_base = [str(p) for p in required_base if not p.exists()]
    if missing_base:
        raise RuntimeError(f"BLIP base assets missing: {missing_base}")

    vocab_path = blip_dir / "vocab.txt"
    if not vocab_path.exists():
        vocab_path = _build_vocab_from_tokenizer_json(blip_dir)
        print(f"[REPAIR] wrote {vocab_path}")
    else:
        print(f"[REPAIR] already exists: {vocab_path}")

    print("[REPAIR] BLIP strict assets ready.")


if __name__ == "__main__":
    main()
