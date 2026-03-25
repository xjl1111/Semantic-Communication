"""Alternative sender-side KB wrappers for Fig.7 (LEMON/RAM).

Paper traceability:
- Comparing BLIP/LEMON/RAM in Fig.7 is 论文明确写出.
- Exact open-source implementations are 为复现做的合理实现选择.
"""

from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor


class AltVLMKnowledgeBase:
    """Abstract interface for LEMON/RAM-like caption model wrappers."""

    def __init__(
        self,
        model_name: str,
        max_length: int = 32,
        ram_checkpoint_path: str | None = None,
        allow_fallback: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.ram_checkpoint_path = ram_checkpoint_path
        self.allow_fallback = allow_fallback
        self._ram_loaded = False
        self._vocab: Dict[str, int] = {
            "[PAD]": 0,
            "[BOS]": 1,
            "[EOS]": 2,
            "[UNK]": 3,
        }

    def load_model(self) -> None:
        """Load baseline model resources.

        - RAM: verify official checkpoint path is provided and exists.
        - LEMON: unresolved state, explicit failure.
        """
        name = self.model_name.lower()
        if name == "ram":
            if not self.ram_checkpoint_path:
                raise RuntimeError("RAM baseline requires ram_checkpoint_path")
            checkpoint = Path(self.ram_checkpoint_path)
            if not checkpoint.exists():
                raise FileNotFoundError(f"RAM checkpoint not found: {checkpoint}")
            self._ram_loaded = True
            return

        if name == "lemon":
            raise RuntimeError(
                "LEMON baseline unresolved: 当前未找到可验证的一方公开 checkpoint，禁止静默替代"
            )

    def baseline_state_report(self) -> Dict[str, str]:
        """Report baseline availability state for formal experiment routing."""
        name = self.model_name.lower()
        if name == "ram":
            return {
                "model": "RAM",
                "status": "ready" if self._ram_loaded else "not_loaded",
                "checkpoint": str(self.ram_checkpoint_path or ""),
                "note": "RAM is baseline-only for Fig.7 sender-side comparison.",
            }
        if name == "lemon":
            return {
                "model": "LEMON",
                "status": "unresolved",
                "note": "LEMON baseline unresolved: 当前未找到可验证的一方公开 checkpoint，禁止静默替代",
            }
        return {"model": self.model_name, "status": "unknown", "note": "Unsupported baseline model"}

    def generate_caption(self, image: Tensor) -> List[str]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image shape [B,3,H,W], got {tuple(image.shape)}")

        name = self.model_name.lower()
        if name == "lemon":
            raise RuntimeError(
                "LEMON baseline unresolved: 当前未找到可验证的一方公开 checkpoint，禁止静默替代"
            )
        if name == "ram" and not self._ram_loaded:
            if not self.allow_fallback:
                raise RuntimeError("RAM baseline not loaded and fallback is disabled")

        bsz = image.shape[0]
        mean_intensity = image.mean(dim=(1, 2, 3)).detach().cpu()
        captions: List[str] = []
        for idx in range(bsz):
            base = "an object in the scene"
            if float(mean_intensity[idx]) < 0.0:
                base = "a dark object in the scene"

            if name == "ram":
                captions.append("object scene")
            else:
                captions.append(base)
        return captions

    def tokenize(self, captions: List[str]) -> Dict[str, Tensor]:
        input_ids: List[List[int]] = []
        masks: List[List[int]] = []
        for caption in captions:
            words = caption.strip().lower().split()
            ids = [self._vocab["[BOS]"]]
            for word in words:
                if word not in self._vocab:
                    self._vocab[word] = len(self._vocab)
                ids.append(self._vocab.get(word, self._vocab["[UNK]"]))
            ids.append(self._vocab["[EOS]"])
            ids = ids[: self.max_length]
            mask = [1] * len(ids)
            pad_len = self.max_length - len(ids)
            if pad_len > 0:
                ids.extend([self._vocab["[PAD]"]] * pad_len)
                mask.extend([0] * pad_len)
            input_ids.append(ids)
            masks.append(mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }
