"""Verify Stable Diffusion loading and text-to-image reconstruction.

Usage:
  python scripts/verify_sd.py --prompt "a cat on a chair"
"""

from __future__ import annotations

import argparse
import traceback

import torch
from diffusers import StableDiffusionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="sd-legacy/stable-diffusion-v1-5")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="D:/model_cache/vlm_csc/sd")
    parser.add_argument("--output", type=str, default="outputs/verify_sd.png")
    parser.add_argument("--steps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[verify_sd] model={args.model_name}")
    print(f"[verify_sd] device={device}")
    print(f"[verify_sd] cache_dir={args.cache_dir}")

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            torch_dtype=dtype,
        )
        pipe = pipe.to(device)

        generator = torch.Generator(device=device)
        generator.manual_seed(42)
        result = pipe(args.prompt, num_inference_steps=args.steps, generator=generator)
        image = result.images[0]
        image.save(args.output)

        print("[verify_sd] success=True")
        print(f"[verify_sd] output_image={args.output}")
        print(f"[verify_sd] prompt={args.prompt}")
    except Exception as exc:
        print("[verify_sd] success=False")
        print(f"[verify_sd] error_type={type(exc).__name__}")
        print(f"[verify_sd] error={exc}")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
