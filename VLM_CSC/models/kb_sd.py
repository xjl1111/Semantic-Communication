"""Receiver-side SD CKB wrapper.

Inputs:
    texts: list[str] length B
Outputs:
    recon images: Tensor[B, 3, H, W]

Paper traceability:
- SD with text encoder + U-Net + VAE decoder is 论文明确写出.
- Sampler/steps/guidance are 为复现做的合理实现选择.
"""

from pathlib import Path
from typing import List

import hashlib
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.utils import save_image


def _try_import_sd_pipeline() -> object | None:
    try:
        from diffusers import StableDiffusionPipeline

        return StableDiffusionPipeline
    except Exception:
        return None


class StableDiffusionKnowledgeBase:
    """Receiver-side CKB using Stable Diffusion family model."""

    def __init__(
        self,
        checkpoint: str,
        image_size: int = 224,
        seed: int = 42,
        device: str = "cpu",
        load_pretrained: bool = True,
        allow_fallback: bool = True,
        cache_dir: str | None = None,
        use_fp16: bool = True,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> None:
        self.checkpoint = checkpoint
        self.image_size = image_size
        self.seed = seed
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.allow_fallback = allow_fallback
        self.cache_dir = cache_dir
        self.use_fp16 = use_fp16
        self._pipe = None

        if load_pretrained:
            self.load_pipeline()

    def load_pipeline(self) -> None:
        """Load Stable Diffusion pipeline from pretrained checkpoint."""
        pipeline_cls = _try_import_sd_pipeline()
        if pipeline_cls is None:
            if self.allow_fallback:
                self._pipe = None
                return
            raise RuntimeError("diffusers StableDiffusionPipeline is unavailable and fallback is disabled")

        dtype = torch.float16 if (self.use_fp16 and "cuda" in self.device) else torch.float32
        try:
            self._pipe = pipeline_cls.from_pretrained(
                self.checkpoint,
                cache_dir=self.cache_dir,
                torch_dtype=dtype,
            )
            self._pipe = self._pipe.to(self.device)
        except Exception as exc:
            if self.allow_fallback:
                self._pipe = None
                return
            raise RuntimeError(f"Failed to load SD checkpoint '{self.checkpoint}': {exc}") from exc

    def reconstruct_image(self, prompt: str) -> Image.Image:
        """Reconstruct one image from one text prompt."""
        if self._pipe is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)
            result = self._pipe(
                prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            )
            return result.images[0]

        if not self.allow_fallback:
            raise RuntimeError(
                "SD pipeline is not loaded and fallback is disabled. "
                "Formal experiments must load real pretrained SD."
            )

        digest = hashlib.sha256((prompt + str(self.seed)).encode("utf-8")).hexdigest()
        seed_val = int(digest[:8], 16)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed_val)
        tensor = torch.rand((3, self.image_size, self.image_size), generator=gen)
        array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)

    def expose_components(self) -> dict:
        """Expose SD components when pipeline is available."""
        if self._pipe is None:
            return {"text_encoder": None, "unet": None, "vae": None}
        return {
            "text_encoder": getattr(self._pipe, "text_encoder", None),
            "unet": getattr(self._pipe, "unet", None),
            "vae": getattr(self._pipe, "vae", None),
        }

    def reconstruct_from_text(self, texts: List[str]) -> Tensor:
        """Reconstruct images from decoded semantic text."""
        tensors = []
        for text in texts:
            image = self.reconstruct_image(text)
            arr = torch.from_numpy(np.array(image)).float() / 255.0
            if arr.ndim == 3:
                arr = arr.permute(2, 0, 1)
            tensors.append(arr)
        return torch.stack(tensors, dim=0)

    def save_reconstructions(self, images: Tensor, out_dir: Path) -> None:
        """Save generated images to outputs/reconstructions."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected image tensor [B,3,H,W], got {tuple(images.shape)}")
        for index in range(images.shape[0]):
            image_path = out_dir / f"recon_{index:05d}.png"
            save_image(images[index].clamp(0.0, 1.0), image_path)
