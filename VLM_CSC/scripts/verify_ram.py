"""Verify RAM baseline loading/inference using official recognize-anything script.

Usage:
  python scripts/verify_ram.py \
    --repo D:/third_party/recognize-anything \
    --image D:/data/test.jpg \
    --checkpoint D:/third_party/recognize-anything/pretrained/ram_swin_large_14m.pth
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True, help="recognize-anything repository root")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo)
    image = Path(args.image)
    checkpoint = Path(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[verify_ram] model=RAM")
    print(f"[verify_ram] device={device}")
    print(f"[verify_ram] repo={repo}")
    print(f"[verify_ram] checkpoint={checkpoint}")

    try:
        script = repo / "inference_ram.py"
        if not script.exists():
            raise FileNotFoundError(f"inference_ram.py not found: {script}")
        if not image.exists():
            raise FileNotFoundError(f"image not found: {image}")
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

        cmd = [
            sys.executable,
            str(script),
            "--image",
            str(image),
            "--pretrained",
            str(checkpoint),
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        )

        print("[verify_ram] success=True")
        print("[verify_ram] stdout_tail=")
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        for ln in lines[-10:]:
            print(f"  {ln}")
    except Exception as exc:
        print("[verify_ram] success=False")
        print(f"[verify_ram] error_type={type(exc).__name__}")
        print(f"[verify_ram] error={exc}")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
