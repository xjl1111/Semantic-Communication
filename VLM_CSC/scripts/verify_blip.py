"""Verify BLIP loading and caption inference.

Usage:
  python scripts/verify_blip.py --image path/to/test.jpg
"""

from __future__ import annotations

import argparse
import traceback

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="D:/model_cache/vlm_csc/blip")
    parser.add_argument("--max-length", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[verify_blip] model={args.model_name}")
    print(f"[verify_blip] device={device}")
    print(f"[verify_blip] cache_dir={args.cache_dir}")

    try:
        processor = BlipProcessor.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        model = BlipForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        model = model.to(device)
        model.eval()

        image = Image.open(args.image).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=args.max_length)

        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        tokens = processor.tokenizer(
            [caption],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )

        print("[verify_blip] success=True")
        print(f"[verify_blip] caption={caption}")
        print(f"[verify_blip] token_ids_shape={tuple(tokens['input_ids'].shape)}")
        print(f"[verify_blip] attention_mask_shape={tuple(tokens['attention_mask'].shape)}")
    except Exception as exc:
        print("[verify_blip] success=False")
        print(f"[verify_blip] error_type={type(exc).__name__}")
        print(f"[verify_blip] error={exc}")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
