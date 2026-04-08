"""BLIP-based sender CKB module."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image

from model.preprocessing import ImageUpsampler
from model.senders.base import SenderCKB


class SenderCKB_BLIP(SenderCKB):
    """BLIP image captioning sender with configurable super-resolution.

    caption_mode:
        "sr"       - SR + BLIP-base + prompt="a photo of a"
        "sr_prompt" - SR + BLIP-base + prompt="a photo of a"
        "prompt"   - No SR + BLIP-base + prompt="a photo of a"
        "blip2"    - SR + BLIP-2 (blip2-opt-2.7b), no prompt

    caption_prompt:
        Custom prompt text (overrides caption_mode default)
        None = use caption_mode default prompt
    """

    _DEFAULT_PROMPTS: Dict[str, str | None] = {
        "sr":        "a photo of a",
        "sr_prompt": "a photo of a",
        "prompt":    "a photo of a",
        "blip2":     None,
        "baseline":  "a photo of a",
    }

    _SR_ENABLED_MODES = frozenset({"sr", "sr_prompt", "blip2", "baseline"})
    _VALID_CAPTION_MODES = ("sr", "sr_prompt", "prompt", "blip2", "baseline")

    def __init__(
        self,
        blip_dir: str | Path,
        use_real_ckb: bool = False,
        device: str = "cpu",
        caption_mode: str = "baseline",
        caption_prompt: str | None = None,
    ):
        self.use_real_ckb = bool(use_real_ckb)
        self.device = device
        self.blip_dir = str(blip_dir)

        if caption_mode not in self._VALID_CAPTION_MODES:
            raise ValueError(
                f"caption_mode must be one of {self._VALID_CAPTION_MODES}, got '{caption_mode}'"
            )
        self.caption_mode = caption_mode

        self.upsampler = ImageUpsampler(
            target_size=256,
            enabled=(caption_mode in self._SR_ENABLED_MODES),
        )

        if caption_prompt is None:
            self.caption_prompt = self._DEFAULT_PROMPTS.get(caption_mode, "a photo of a")
        elif caption_prompt == "":
            self.caption_prompt = None
        else:
            self.caption_prompt: str | None = str(caption_prompt)

        self.processor: Any = None
        self.model: Any = None

        if self.use_real_ckb:
            if self.caption_mode == "blip2":
                self._init_blip2()
            else:
                self._init_blip_base()

    def _init_blip_base(self) -> None:
        """Load local BLIP-base model."""
        from transformers import BlipForConditionalGeneration, BlipProcessor

        blip_path = Path(self.blip_dir)
        required_files = [
            blip_path / "config.json",
            blip_path / "model.safetensors",
            blip_path / "preprocessor_config.json",
            blip_path / "tokenizer_config.json",
            blip_path / "vocab.txt",
        ]
        missing = [str(p) for p in required_files if not p.exists()]
        if missing:
            raise RuntimeError(
                "BLIP local assets incomplete for strict mode (use_fast=False). Missing: "
                f"{missing}. Please run VLM_CSC/experiments/tools/repair_blip_assets.py before training/evaluation."
            )

        # Handle transformers version compatibility
        try:
            self.processor = BlipProcessor.from_pretrained(
                self.blip_dir,
                use_fast=False,
                local_files_only=True,
            )
        except TypeError as e:
            # Newer transformers versions may have config conflicts
            if "image_processor" in str(e):
                from transformers import BlipImageProcessor, BertTokenizerFast
                image_processor = BlipImageProcessor.from_pretrained(
                    self.blip_dir, local_files_only=True
                )
                tokenizer = BertTokenizerFast.from_pretrained(
                    self.blip_dir, local_files_only=True
                )
                self.processor = BlipProcessor(image_processor=image_processor, tokenizer=tokenizer)
            else:
                raise

        self.model = BlipForConditionalGeneration.from_pretrained(
            self.blip_dir,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _init_blip2(self) -> None:
        """Load BLIP-2 (Salesforce/blip2-opt-2.7b) model."""
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        _model_id = "Salesforce/blip2-opt-2.7b"
        print(f"[SenderCKB_BLIP] Loading BLIP-2 ({_model_id}) ...")
        self.processor = Blip2Processor.from_pretrained(_model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            _model_id, dtype=torch.float16,
        )
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        print("[SenderCKB_BLIP] BLIP-2 loaded successfully.")

    @staticmethod
    def _detect_degenerate_caption(text: str, max_repeat: int = 3) -> bool:
        """Detect degenerate captions with excessive word repetition."""
        words = text.lower().split()
        if len(words) <= 4:
            return False
        wc = Counter(words)
        most_common_word, most_common_count = wc.most_common(1)[0]
        if most_common_count > max_repeat and most_common_count / len(words) > 0.35:
            return True
        for n in (2, 3, 4):
            if len(words) < n * 2:
                continue
            ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
            ngram_counts = Counter(ngrams)
            top_ng, top_count = ngram_counts.most_common(1)[0]
            if top_count >= 3 and (top_count * n) / len(words) > 0.4:
                return True
        return False

    @torch.inference_mode()
    def forward(self, image: Image.Image | torch.Tensor) -> str:
        if not self.use_real_ckb:
            return "a bird standing on a branch"

        pil_image: Image.Image
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            np_arr = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
            pil_image = Image.fromarray(np_arr)
        else:
            pil_image = image

        pil_image = pil_image.convert("RGB")
        pil_image = self.upsampler(pil_image)
        prompt_text: str | None = self.caption_prompt

        assert self.processor is not None and self.model is not None

        if self.caption_mode == "blip2":
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device, torch.float16) for k, v in inputs.items()}
        else:
            inputs = self.processor(images=pil_image, text=prompt_text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        _bad_words_ids = [self.processor.tokenizer.encode(w, add_special_tokens=False)
                          for w in ["and"]]
        output_ids = self.model.generate(
            **inputs, max_new_tokens=64, num_beams=4,
            bad_words_ids=_bad_words_ids,
        )
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()

        if self._detect_degenerate_caption(caption):
            output_ids_retry = self.model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=1,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=2.0,
            )
            caption_retry = self.processor.decode(output_ids_retry[0], skip_special_tokens=True).strip()
            if not self._detect_degenerate_caption(caption_retry) and len(caption_retry) > 3:
                caption = caption_retry
            else:
                words = caption.split()
                seen = set()
                trimmed = []
                for w in words:
                    wl = w.lower()
                    if wl in seen and len(trimmed) > 3:
                        break
                    seen.add(wl)
                    trimmed.append(w)
                caption = " ".join(trimmed)

        return caption
