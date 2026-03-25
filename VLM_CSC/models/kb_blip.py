"""BLIP sender-side CKB wrapper.

Inputs:
    image: Tensor[B, 3, H, W]
Outputs:
    captions: list[str]
    token ids: Tensor[B, T]
    attention mask: Tensor[B, T]

Paper traceability:
- BLIP-based sender CKB is 论文明确写出.
- Specific checkpoint and max T are 为复现做的合理实现选择.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor
from torchvision.transforms.functional import to_pil_image


def _try_import_blip() -> tuple[object, object] | tuple[None, None]:
    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        return BlipProcessor, BlipForConditionalGeneration
    except Exception:
        return None, None


class BlipKnowledgeBase:
    """Sender-side cross-modal knowledge base based on BLIP."""

    def __init__(
        self,
        checkpoint: str,
        max_length: int = 32,
        device: str = "cpu",
        load_pretrained: bool = True,
        allow_fallback: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.max_length = max_length
        self.device = device
        self.allow_fallback = allow_fallback
        self.cache_dir = cache_dir
        self._processor = None
        self._model = None

        self._vocab: Dict[str, int] = {
            "[PAD]": 0,
            "[BOS]": 1,
            "[EOS]": 2,
            "[UNK]": 3,
        }

        if load_pretrained:
            self.load_model()

    def load_model(self) -> None:
        """Load BLIP processor/model from pretrained checkpoint."""
        processor_cls, model_cls = _try_import_blip()
        if processor_cls is None or model_cls is None:
            if self.allow_fallback:
                self._processor = None
                self._model = None
                return
            raise RuntimeError("transformers BLIP dependencies are unavailable and fallback is disabled")

        try:
            self._processor = processor_cls.from_pretrained(self.checkpoint, cache_dir=self.cache_dir)
            self._model = model_cls.from_pretrained(self.checkpoint, cache_dir=self.cache_dir).to(self.device)
            self._model.eval()
        except Exception as exc:
            if self.allow_fallback:
                self._processor = None
                self._model = None
                return
            raise RuntimeError(f"Failed to load BLIP checkpoint '{self.checkpoint}': {exc}") from exc

    def generate_caption(self, image: Tensor) -> List[str]:
        """Generate captions for image tensor [B,3,H,W]."""
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image shape [B,3,H,W], got {tuple(image.shape)}")

        if self._processor is not None and self._model is not None:
            images = [to_pil_image(img.detach().cpu().clamp(0.0, 1.0)) for img in image]
            inputs = self._processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self._model.generate(**inputs, max_length=self.max_length)
            captions = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
            return [caption.strip() for caption in captions]

        if not self.allow_fallback:
            raise RuntimeError(
                "BLIP model is not loaded and fallback is disabled. "
                "Formal experiments must load real pretrained BLIP."
            )

        batch_size = image.shape[0]
        captions: List[str] = []
        mean_intensity = image.mean(dim=(1, 2, 3)).detach().cpu()
        for idx in range(batch_size):
            if float(mean_intensity[idx]) > 0.0:
                captions.append("an object in the scene")
            else:
                captions.append("a dark object in the scene")
        return captions

    def tokenize(self, captions: List[str]) -> Dict[str, Tensor]:
        """Tokenize captions into ids and attention mask tensors [B,T]."""
        if self._processor is not None:
            tokens = self._processor.tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": tokens["input_ids"].long(),
                "attention_mask": tokens["attention_mask"].long(),
            }

        if not self.allow_fallback:
            raise RuntimeError("BLIP tokenizer unavailable and fallback is disabled")

        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []

        for caption in captions:
            words = caption.strip().lower().split()
            token_ids = [self._vocab["[BOS]"]]
            for word in words:
                if word not in self._vocab:
                    self._vocab[word] = len(self._vocab)
                token_ids.append(self._vocab.get(word, self._vocab["[UNK]"]))
            token_ids.append(self._vocab["[EOS]"])
            token_ids = token_ids[: self.max_length]
            mask = [1] * len(token_ids)

            pad_len = self.max_length - len(token_ids)
            if pad_len > 0:
                token_ids = token_ids + [self._vocab["[PAD]"]] * pad_len
                mask = mask + [0] * pad_len

            input_ids.append(token_ids)
            attention_mask.append(mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def generate_caption_tokens(self, image: Tensor) -> Dict[str, object]:
        """Generate captions and token tensors in one call.

        Returns:
            {
                "captions": List[str],
                "input_ids": Tensor[B,T],
                "attention_mask": Tensor[B,T],
            }
        """
        captions = self.generate_caption(image)
        tokens = self.tokenize(captions)
        return {
            "captions": captions,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }
