"""RAM-based sender CKB module."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vlm_csc.senders.base import SenderCKB


class SenderCKB_RAM(SenderCKB):
    """RAM (Recognize Anything Model) sender for tag-based descriptions."""

    def __init__(
        self,
        ram_ckpt: str | Path,
        use_real_ckb: bool = False,
        device: str = "cpu",
        image_size: int = 384,
    ):
        self.use_real_ckb = bool(use_real_ckb)
        self.device = device
        self.ram_ckpt = str(ram_ckpt)
        self.image_size = int(image_size)

        self.model: Any = None
        self.transform: Any = None
        self._inference_ram: Any = None

        if self.use_real_ckb:
            import importlib

            _this_dir = Path(__file__).resolve().parent
            candidate_roots = [
                _this_dir.parent.parent.parent.parent / "data" / "assets" / "downloaded_models" / "recognize-anything",
                _this_dir / "ckb_models" / "recognize-anything",
            ]
            for root in candidate_roots:
                root_str = str(root.resolve())
                if root.exists() and root_str not in sys.path:
                    sys.path.insert(0, root_str)

            ram_pkg = importlib.import_module("ram")
            ram_models_pkg = importlib.import_module("ram.models")
            ram_ctor = getattr(ram_models_pkg, "ram")
            get_transform = getattr(ram_pkg, "get_transform")
            inference_ram = getattr(ram_pkg, "inference_ram")

            self.transform = get_transform(image_size=self.image_size)
            self.model = ram_ctor(pretrained=self.ram_ckpt, image_size=self.image_size, vit="swin_l")
            self.model = self.model.to(self.device).eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self._inference_ram = inference_ram

    @staticmethod
    def _to_pil(image: Image.Image | torch.Tensor) -> Image.Image:
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            np_arr = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
            return Image.fromarray(np_arr).convert("RGB")
        return image.convert("RGB")

    @torch.inference_mode()
    def forward(self, image: Image.Image | torch.Tensor) -> str:
        if not self.use_real_ckb:
            return "This image contains: dog, animal, pet."

        image = self._to_pil(image)
        assert self.transform is not None and self._inference_ram is not None
        x = self.transform(image).unsqueeze(0).to(self.device)
        eng_tags, _ = self._inference_ram(x, self.model)

        if isinstance(eng_tags, (list, tuple)):
            eng_tags = eng_tags[0]
        raw_tags = str(eng_tags).strip()
        tags = [tag.strip().lower() for tag in raw_tags.replace("|", ",").split(",") if tag.strip()]
        tags = list(dict.fromkeys(tags))
        tags = [tag for tag in tags if len(tag) > 1]
        if len(tags) == 0:
            return "This image contains: object."
        return "This image contains: " + ", ".join(tags) + "."
