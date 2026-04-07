"""Asset download utilities for VLM-CSC models."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def download_clip(models: list[str] | None = None) -> None:
    """Download CLIP classifier models to local HuggingFace cache.

    Args:
        models: List of model IDs to download. Defaults to ["vitl14"].
    """
    _MODELS = {
        "vitl14": "openai/clip-vit-large-patch14",
        "vitb32": "openai/clip-vit-base-patch32",
    }

    if models is None:
        models = ["vitl14"]

    for model_key in models:
        model_name = _MODELS.get(model_key, model_key)
        print(f"[download_clip] Downloading: {model_name} ...")
        try:
            from transformers import CLIPModel, CLIPProcessor

            CLIPProcessor.from_pretrained(model_name)
            CLIPModel.from_pretrained(model_name)
            print(f"[download_clip] Done: {model_name}")
        except Exception as exc:
            print(f"[download_clip] Failed: {model_name}: {exc}", file=sys.stderr)
            raise


def download_ram_weight(local_dir: Path | str | None = None) -> str:
    """Download RAM model weight from HuggingFace.

    Args:
        local_dir: Local directory to save the weight. Defaults to data/assets/downloaded_models/ram/pretrained.

    Returns:
        Path to the downloaded file.
    """
    from huggingface_hub import hf_hub_download

    if local_dir is None:
        local_dir = Path(__file__).resolve().parents[3] / "data" / "assets" / "downloaded_models" / "ram" / "pretrained"
    else:
        local_dir = Path(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(
        repo_id="xinyu1205/recognize-anything",
        repo_type="space",
        filename="ram_swin_large_14m.pth",
        local_dir=str(local_dir),
    )
    print(f"[download_ram_weight] Downloaded: {path}")
    return path


def download_sd_tokenizer_files(local_dir: Path | str | None = None) -> list[str]:
    """Download Stable Diffusion tokenizer files.

    Args:
        local_dir: Local directory to save the files. Defaults to data/assets/downloaded_models/sd15/tokenizer.

    Returns:
        List of paths to downloaded files.
    """
    from huggingface_hub import hf_hub_download

    repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    if local_dir is None:
        local_dir = Path(__file__).resolve().parents[3] / "data" / "assets" / "downloaded_models" / "sd15" / "tokenizer"
    else:
        local_dir = Path(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for filename in ["vocab.json", "merges.txt", "special_tokens_map.json"]:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=f"tokenizer/{filename}",
            local_dir=str(local_dir.parent),
            local_dir_use_symlinks=False,
        )
        print(f"[download_sd_tokenizer] Downloaded: {path}")
        paths.append(path)

    return paths


def main() -> None:
    """CLI entry point for downloading all assets."""
    parser = argparse.ArgumentParser(description="Download VLM-CSC model assets")
    parser.add_argument("--clip", action="store_true", help="Download CLIP models")
    parser.add_argument("--clip-all", action="store_true", help="Download all CLIP variants")
    parser.add_argument("--ram", action="store_true", help="Download RAM weight")
    parser.add_argument("--sd-tokenizer", action="store_true", help="Download SD tokenizer files")
    parser.add_argument("--all", action="store_true", help="Download all assets")
    args = parser.parse_args()

    if args.all or args.clip or args.clip_all:
        models = ["vitl14", "vitb32"] if (args.all or args.clip_all) else ["vitl14"]
        download_clip(models)

    if args.all or args.ram:
        download_ram_weight()

    if args.all or args.sd_tokenizer:
        download_sd_tokenizer_files()

    print("[download] All requested assets downloaded.")


if __name__ == "__main__":
    main()
