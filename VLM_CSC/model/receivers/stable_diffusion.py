"""Stable Diffusion-based receiver CKB module."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from model.receivers.base import ReceiverCKB


class ReceiverCKB_SD(ReceiverCKB):
    """Stable Diffusion 1.5 receiver for text-to-image generation."""

    def __init__(self, sd_dir: str | Path, use_real_ckb: bool = False, device: str = "cpu"):
        self.use_real_ckb = bool(use_real_ckb)
        self.device = device
        self.sd_dir = str(sd_dir)

        self.pipe = None
        if self.use_real_ckb:
            from diffusers import StableDiffusionPipeline

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.pipe = StableDiffusionPipeline.from_pretrained(self.sd_dir, torch_dtype=dtype)
            try:
                self.pipe.safety_checker = None
                self.pipe.feature_extractor = None
                if hasattr(self.pipe, "requires_safety_checker"):
                    self.pipe.requires_safety_checker = False
            except Exception:
                pass
            self.pipe = self.pipe.to(self.device)
            try:
                self.pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass
            if self.device == "cuda":
                try:
                    self.pipe.enable_attention_slicing()
                except Exception:
                    pass

    @torch.inference_mode()
    def forward(
        self,
        text: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 42,
    ) -> Image.Image:
        if not self.use_real_ckb:
            return Image.new("RGB", (256, 256), color=(200, 200, 200))

        generator = torch.Generator(device=self.device).manual_seed(int(seed))

        result = self.pipe(
            prompt=text,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]
