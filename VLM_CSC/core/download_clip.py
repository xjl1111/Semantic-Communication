"""
下载 CLIP 分类器模型到本地 HuggingFace 缓存。

用法：
    python download_clip.py               # 下载 ViT-L/14（推荐，~430M）
    python download_clip.py --all         # 下载 ViT-L/14 + ViT-B/32（~580M）
    python download_clip.py --vitb32_only # 仅下载 ViT-B/32（~150M，较快）

下载完成后评估时将自动优先使用 ViT-L/14。
"""
from __future__ import annotations

import argparse
import sys


_MODELS = {
    "vitl14": "openai/clip-vit-large-patch14",
    "vitb32": "openai/clip-vit-base-patch32",
}


def _download(model_name: str) -> None:
    print(f"[download_clip] 正在下载: {model_name} …")
    try:
        from transformers import CLIPModel, CLIPProcessor

        CLIPProcessor.from_pretrained(model_name)
        CLIPModel.from_pretrained(model_name)
        print(f"[download_clip] ✓ 完成: {model_name}")
    except Exception as exc:
        print(f"[download_clip] ✗ 失败: {model_name}: {exc}", file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 CLIP 模型到本地缓存")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all",         action="store_true", help="下载所有变体 (ViT-L/14 + ViT-B/32)")
    group.add_argument("--vitb32_only", action="store_true", help="仅下载 ViT-B/32（体积较小，约 150M）")
    args = parser.parse_args()

    if args.vitb32_only:
        to_download = [_MODELS["vitb32"]]
    elif args.all:
        to_download = [_MODELS["vitl14"], _MODELS["vitb32"]]
    else:
        # 默认只下载最佳模型
        to_download = [_MODELS["vitl14"]]

    for m in to_download:
        _download(m)

    print("[download_clip] 全部完成。运行 eval_figN.py 时将自动使用这些缓存。")


if __name__ == "__main__":
    main()
