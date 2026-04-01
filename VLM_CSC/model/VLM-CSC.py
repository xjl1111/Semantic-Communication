from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter


# ═══════════════════════════════════════════════════════════════════════════════
# ImageUpsampler — 模型内置 LANCZOS 超分辨率上采样器
# ═══════════════════════════════════════════════════════════════════════════════

class ImageUpsampler:
    """LANCZOS 超分辨率 + UnsharpMask 锐化，作为模型内置组件。

    数据集图像通常为 32×32，BLIP/BLIP-2 期望 ≥256px 输入。
    LANCZOS 插值 + UnsharpMask 在不引入额外模型的前提下
    将 caption Level-A 准确率从 ~73% 提升至 ~83%。

    参数:
        target_size: 上采样目标尺寸（短边，默认 256）
        sharpen_radius: UnsharpMask 半径
        sharpen_percent: UnsharpMask 强度百分比（默认 250，强锐化）
        sharpen_threshold: UnsharpMask 阈值（默认 2，低阈值增强细节）
        enabled: 是否启用上采样（默认 True）
    """

    def __init__(
        self,
        target_size: int = 256,
        sharpen_radius: float = 2.0,
        sharpen_percent: int = 250,
        sharpen_threshold: int = 2,
        enabled: bool = True,
    ):
        self.target_size = int(target_size)
        self.sharpen_radius = float(sharpen_radius)
        self.sharpen_percent = int(sharpen_percent)
        self.sharpen_threshold = int(sharpen_threshold)
        self.enabled = bool(enabled)

    def __call__(self, pil_image: Image.Image) -> Image.Image:
        """对 PIL 图像执行上采样 + 锐化。"""
        if not self.enabled:
            return pil_image

        w, h = pil_image.size
        if w >= self.target_size and h >= self.target_size:
            return pil_image

        scale = max(self.target_size / w, self.target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        upscaled = pil_image.resize((new_w, new_h), Image.LANCZOS)  # type: ignore[attr-defined]
        sharpened = upscaled.filter(
            ImageFilter.UnsharpMask(
                radius=self.sharpen_radius,
                percent=self.sharpen_percent,
                threshold=self.sharpen_threshold,
            )
        )
        return sharpened

    def __repr__(self) -> str:
        return (
            f"ImageUpsampler(target_size={self.target_size}, "
            f"sharpen_radius={self.sharpen_radius}, "
            f"sharpen_percent={self.sharpen_percent}, "
            f"sharpen_threshold={self.sharpen_threshold}, "
            f"enabled={self.enabled})"
        )

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from communication_modules import SemanticEncoder
from communication_modules import SemanticDecoder
from communication_modules import ChannelEncoder
from communication_modules import ChannelDecoder
from communication_modules import PhysicalChannel
from med import MED


class SenderCKB_BLIP:
    """BLIP 侧图像描述生成器，可配置超分辨率上采样。

    核心组件:
        upsampler:  ImageUpsampler — LANCZOS 超分辨率 + UnsharpMask 锐化
                    32×32 → 256×256。根据 caption_mode 决定是否启用。

    caption_mode:
        \"sr\"       — SR + BLIP-base + prompt=\"a photo of a\"
        \"sr_prompt\" — SR + BLIP-base + prompt=\"a photo of a\"（中性 prompt，避免混合 caption）
        \"prompt\"   — 无 SR + BLIP-base + prompt=\"a photo of a\"（中性 prompt）
        \"blip2\"    — SR + BLIP-2 (blip2-opt-2.7b), 无 prompt

    caption_prompt:
        自定义 prompt 文本（覆盖 caption_mode 默认值）
        None = 按 caption_mode 的默认 prompt
    """

    # 默认 prompt 映射（按 caption_mode 决定缺省值）
    _DEFAULT_PROMPTS: Dict[str, str | None] = {
        "sr":        "a photo of a",
        "sr_prompt": "a photo of a",   # 中性 prompt，避免引导 BLIP 同时生成 cat+dog
        "prompt":    "a photo of a",   # 同上；"a photo of an animal, a" 会导致混合 caption
        "blip2":     None,  # BLIP-2 不使用文本前缀
        # 向后兼容旧 checkpoint（baseline 等同于 sr）
        "baseline":  "a photo of a",
    }

    # 需要启用 SR 的模式
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

        # ── 上采样器（根据模式决定是否启用）────────────────────────────────
        self.upsampler = ImageUpsampler(
            target_size=256,
            enabled=(caption_mode in self._SR_ENABLED_MODES),
        )

        # ── caption prompt（可配置，None 表示使用 caption_mode 默认值） ─────
        if caption_prompt is not None:
            self.caption_prompt: str | None = str(caption_prompt)
        else:
            self.caption_prompt = self._DEFAULT_PROMPTS.get(caption_mode, "a photo of a")

        self.processor: Any = None
        self.model: Any = None

        if self.use_real_ckb:
            if self.caption_mode == "blip2":
                self._init_blip2()
            else:
                self._init_blip_base()

    # ── 模型初始化 ─────────────────────────────────────────────────────────

    def _init_blip_base(self) -> None:
        """加载本地 BLIP-base 模型（baseline / sr_prompt 模式共用）。"""
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
                f"{missing}. Please run VLM_CSC/exp/tools/repair_blip_assets.py before training/evaluation."
            )

        self.processor = BlipProcessor.from_pretrained(
            self.blip_dir,
            use_fast=False,
            local_files_only=True,
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.blip_dir,
            local_files_only=True,
        )
        self.model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _init_blip2(self) -> None:
        """加载 BLIP-2 (Salesforce/blip2-opt-2.7b) 模型。"""
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        _model_id = "Salesforce/blip2-opt-2.7b"
        print(f"[SenderCKB_BLIP] Loading BLIP-2 ({_model_id}) ...")
        self.processor = Blip2Processor.from_pretrained(_model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            _model_id, dtype=torch.float16,
        )
        self.model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        print("[SenderCKB_BLIP] BLIP-2 loaded successfully.")

    # ── 图像预处理 ─────────────────────────────────────────────────────────

    @staticmethod
    def _apply_sr(pil_image: Image.Image, target_size: int = 256) -> Image.Image:
        """[兼容] 旧接口 — 内部已迁移到 ImageUpsampler。"""
        return ImageUpsampler(target_size=target_size)(pil_image)

    # ── 退化检测 ───────────────────────────────────────────────────────────

    @staticmethod
    def _detect_degenerate_caption(text: str, max_repeat: int = 3) -> bool:
        """Detect degenerate captions with excessive word repetition.

        BLIP beam-search sometimes collapses into repetitive text like
        'mouse mouse mouse ...' or 'adopt a cat adopt a cat adopt a cat'.
        These degenerate captions carry almost no useful semantic information
        and will pollute both training and evaluation.
        """
        words = text.lower().split()
        if len(words) <= 4:
            return False
        from collections import Counter
        wc = Counter(words)
        most_common_word, most_common_count = wc.most_common(1)[0]
        # Single-word repetition: "mouse mouse mouse mouse..."
        if most_common_count > max_repeat and most_common_count / len(words) > 0.35:
            return True
        # N-gram repetition: "adopt a cat adopt a cat adopt a cat..."
        # Check if 2-gram or 3-gram patterns repeat excessively
        for n in (2, 3, 4):
            if len(words) < n * 2:
                continue
            ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
            ngram_counts = Counter(ngrams)
            top_ng, top_count = ngram_counts.most_common(1)[0]
            if top_count >= 3 and (top_count * n) / len(words) > 0.4:
                return True
        return False

    # ── 核心推理 ───────────────────────────────────────────────────────────

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

        # ── 内置上采样器（所有模式统一执行）───────────────────────────────
        pil_image = self.upsampler(pil_image)

        # ── 使用已配置的 prompt ──────────────────────────────────────────
        prompt_text: str | None = self.caption_prompt

        assert self.processor is not None and self.model is not None

        # ── 构造模型输入 ──────────────────────────────────────────────────
        if self.caption_mode == "blip2":
            inputs = self.processor(images=pil_image, return_tensors="pt")  # type: ignore[call-arg]
            inputs = {k: v.to(self.device, torch.float16) for k, v in inputs.items()}
        else:
            inputs = self.processor(images=pil_image, text=prompt_text, return_tensors="pt")  # type: ignore[call-arg]
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Primary generation: beam search + ban "and" to prevent mixed captions
        _bad_words_ids = [self.processor.tokenizer.encode(w, add_special_tokens=False)
                          for w in ["and"]]
        output_ids = self.model.generate(
            **inputs, max_new_tokens=64, num_beams=4,
            bad_words_ids=_bad_words_ids,
        )
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()

        # Degeneration guard: if beam-search caption is repetitive, retry with
        # sampling + repetition_penalty to get a cleaner caption.
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
                # Last resort: truncate the repeating tail
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


class SenderCKB_RAM:
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

        self.model = None
        self.transform = None
        self._inference_ram = None

        if self.use_real_ckb:
            import importlib

            candidate_roots = [
                _THIS_DIR.parent / "data" / "assets" / "downloaded_models" / "recognize-anything",
                _THIS_DIR / "ckb_models" / "recognize-anything",
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


class ReceiverCKB_SD:
    def __init__(self, sd_dir: str | Path, use_real_ckb: bool = False, device: str = "cpu"):
        self.use_real_ckb = bool(use_real_ckb)
        self.device = device
        self.sd_dir = str(sd_dir)

        self.pipe = None
        if self.use_real_ckb:
            from diffusers import StableDiffusionPipeline  # type: ignore[attr-defined]

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

        result = self.pipe(  # type: ignore[misc]
            prompt=text,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]  # type: ignore[union-attr]


class SimpleTextTokenizer:
    def __init__(
        self,
        vocab_size: int = 30522,
        pad_id: int = 0,
        bos_id: int = 101,
        eos_id: int = 102,
        tokenizer_dir: str | Path | None = None,
        use_hf_tokenizer: bool = True,
        must_use_hf_tokenizer: bool = True,
    ):
        self.hf_tokenizer = None

        if use_hf_tokenizer:
            try:
                from transformers import AutoTokenizer

                if tokenizer_dir is not None:
                    self.hf_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
                else:
                    self.hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            except Exception as exc:
                if must_use_hf_tokenizer:
                    raise RuntimeError(f"Failed to load HF tokenizer: {exc}") from exc
                self.hf_tokenizer = None

        if must_use_hf_tokenizer and self.hf_tokenizer is None:
            raise RuntimeError("HF tokenizer is required but not available.")

        if self.hf_tokenizer is not None:
            self.vocab_size = int(self.hf_tokenizer.vocab_size)
            self.pad_id = int(self.hf_tokenizer.pad_token_id if self.hf_tokenizer.pad_token_id is not None else 0)

            if self.hf_tokenizer.bos_token_id is not None:
                self.bos_id = int(self.hf_tokenizer.bos_token_id)
            elif self.hf_tokenizer.cls_token_id is not None:
                self.bos_id = int(self.hf_tokenizer.cls_token_id)
            else:
                self.bos_id = int(bos_id)

            if self.hf_tokenizer.eos_token_id is not None:
                self.eos_id = int(self.hf_tokenizer.eos_token_id)
            elif self.hf_tokenizer.sep_token_id is not None:
                self.eos_id = int(self.hf_tokenizer.sep_token_id)
            else:
                self.eos_id = int(eos_id)
        else:
            self.vocab_size = int(vocab_size)
            self.pad_id = int(pad_id)
            self.bos_id = int(bos_id)
            self.eos_id = int(eos_id)

        tokenizer_name = self.hf_tokenizer.__class__.__name__ if self.hf_tokenizer is not None else "SimpleTextTokenizerFallback"
        print(
            f"[TOKENIZER] class={tokenizer_name}, vocab_size={self.vocab_size}, "
            f"bos_id={self.bos_id}, eos_id={self.eos_id}, pad_id={self.pad_id}"
        )

    def encode(self, texts: List[str], max_len: int = 32) -> torch.Tensor:
        if self.hf_tokenizer is not None:
            encoded = self.hf_tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            return encoded["input_ids"].long()

        batch_ids = []
        for text in texts:
            core = [(ord(ch) % (self.vocab_size - 103)) + 103 for ch in text][: max_len - 2]
            ids = [self.bos_id] + core + [self.eos_id]
            ids = ids + [self.pad_id] * (max_len - len(ids))
            batch_ids.append(ids)
        return torch.tensor(batch_ids, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> List[str]:
        if self.hf_tokenizer is not None:
            return self.hf_tokenizer.batch_decode(token_ids, skip_special_tokens=True)

        texts: List[str] = []
        for row in token_ids:
            chars: List[str] = []
            for token in row.tolist():
                if token in (self.pad_id, self.bos_id, self.eos_id):
                    continue
                val = max(32, min(126, (token - 103) % 95 + 32))
                chars.append(chr(val))
            texts.append("".join(chars).strip() if chars else "")
        return texts


class VLMCscSystem(nn.Module):
    """VLM-CSC system model.

    Communication path follows sequence-level transport (seq -> seq):
    semantic token features are transmitted through the channel without sentence-level pooling.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        max_text_len: int = 32,
        vocab_size: int = 30522,
        channel_type: str = "awgn",
        sender_type: str = "blip",
        use_real_ckb: bool = False,
        use_real_receiver_ckb: Optional[bool] = None,
        enable_med: bool = False,
        med_kwargs: Optional[Dict] = None,
        blip_dir: str | Path = "./data/assets/downloaded_models/blip",
        ram_ckpt: str | Path = "./data/assets/downloaded_models/ram_swin_large_14m.pth",
        sd_dir: str | Path = "./data/assets/downloaded_models/sd15",
        device: Optional[str] = None,
        use_nam: bool = True,
        channel_dim: int | None = None,
        caption_mode: str = "baseline",   # baseline / sr_prompt / blip2
        caption_prompt: str | None = None, # 自定义 prompt（None=按 caption_mode 默认值）

    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.channel_dim = int(channel_dim) if channel_dim is not None else self.feature_dim
        self.max_text_len = int(max_text_len)
        self.vocab_size = int(vocab_size)
        self.use_nam = bool(use_nam)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.runtime_device = device
        self.sender_type = str(sender_type).lower()

        if self.sender_type == "blip":
            self.sender_ckb = SenderCKB_BLIP(
                blip_dir=blip_dir,
                use_real_ckb=use_real_ckb,
                device=self.runtime_device,
                caption_mode=caption_mode,
                caption_prompt=caption_prompt,
            )
        elif self.sender_type == "ram":
            self.sender_ckb = SenderCKB_RAM(ram_ckpt=ram_ckpt, use_real_ckb=use_real_ckb, device=self.runtime_device)
        else:
            raise ValueError("sender_type must be 'blip' or 'ram'.")

        if use_real_receiver_ckb is None:
            use_real_receiver_ckb = bool(use_real_ckb)
        self.receiver_ckb = ReceiverCKB_SD(
            sd_dir=sd_dir,
            use_real_ckb=bool(use_real_receiver_ckb),
            device=self.runtime_device,
        )

        self.tokenizer = SimpleTextTokenizer(
            vocab_size=self.vocab_size,
            tokenizer_dir=blip_dir,
            use_hf_tokenizer=True,
            must_use_hf_tokenizer=True,
        )
        self.vocab_size = int(self.tokenizer.vocab_size)
        self.embedding = nn.Embedding(self.vocab_size, self.feature_dim)
        self.register_buffer(
            "pos_encoding",
            self._build_sinusoidal_positional_encoding(self.max_text_len, self.feature_dim),
            persistent=False,
        )

        self.semantic_encoder = SemanticEncoder(feature_dim=self.feature_dim, num_layers=3, num_heads=8, use_nam=self.use_nam)
        self.channel_encoder = ChannelEncoder(input_dim=self.feature_dim, output_dim=self.channel_dim, use_nam=self.use_nam)
        self.channel = PhysicalChannel(channel_type=channel_type)
        self.channel_decoder = ChannelDecoder(input_dim=self.channel_dim, output_dim=self.feature_dim, use_nam=self.use_nam)
        self.semantic_decoder = SemanticDecoder(feature_dim=self.feature_dim, num_layers=3, num_heads=8, use_nam=self.use_nam)
        self.lm_head = nn.Linear(self.feature_dim, self.vocab_size)

        self.enable_med = bool(enable_med)
        self.med = MED(**(med_kwargs or {})) if self.enable_med else None

    @staticmethod
    def _build_sinusoidal_positional_encoding(max_len: int, feature_dim: int) -> torch.Tensor:
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / feature_dim)
        )
        pe = torch.zeros(max_len, feature_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _add_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.max_text_len:
            raise RuntimeError(f"Sequence length {seq_len} exceeds max_text_len={self.max_text_len}")
        pos_enc: torch.Tensor = self.pos_encoding  # type: ignore[assignment]
        return x + pos_enc[:, :seq_len, :].to(device=x.device, dtype=x.dtype)

    @staticmethod
    def _build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def _prepare_decoder_inputs(self, target_ids: torch.Tensor) -> torch.Tensor:
        decoder_input_ids = target_ids.clone()
        decoder_input_ids[:, 1:] = target_ids[:, :-1]
        decoder_input_ids[:, 0] = int(self.tokenizer.bos_id)
        return decoder_input_ids

    @staticmethod
    def _assert_shape_dtype(name: str, x: torch.Tensor, expected_shape: tuple, expected_dtype: Optional[torch.dtype] = None) -> None:
        if tuple(x.shape) != tuple(expected_shape):
            raise RuntimeError(f"{name} shape mismatch: expected {expected_shape}, got {tuple(x.shape)}")
        if expected_dtype is not None and x.dtype != expected_dtype:
            raise RuntimeError(f"{name} dtype mismatch: expected {expected_dtype}, got {x.dtype}")

    def _encode_text(self, texts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        token_ids = self.tokenizer.encode(texts, max_len=self.max_text_len).to(device)
        attention_mask = (token_ids != self.tokenizer.pad_id).long()
        return {"token_ids": token_ids, "attention_mask": attention_mask}

    def _update_med(
        self,
        source_text: str,
        src_ids: torch.Tensor,
        semantic_seq: torch.Tensor,
        src_mask: torch.Tensor,
        image_id: str,
        dataset_id: str,
    ) -> Dict[str, int]:
        if self.med is None:
            return {"triggered": 0, "moved": 0, "stm_size": 0, "ltm_size": 0}

        denom = src_mask.sum(dim=1, keepdim=True).clamp(min=1)
        masked_x = semantic_seq * src_mask.unsqueeze(-1)
        med_feature = masked_x.sum(dim=1) / denom

        self.med.add_to_stm(
            {
                "image_id": image_id,
                "caption_text": source_text,
                "token_ids": src_ids[0],
                "semantic_feature": med_feature[0],
                "dataset_id": dataset_id,
            }
        )
        return self.med.maybe_update()

    def update_med_from_source_text(
        self,
        *,
        source_text: str,
        image_id: str,
        dataset_id: str,
    ) -> Dict[str, int]:
        if self.med is None:
            raise RuntimeError("MED is disabled; update_med_from_source_text is not allowed.")

        device = next(self.parameters()).device
        text_pack = self._encode_text([str(source_text)], device=device)
        src_ids = text_pack["token_ids"]
        src_mask = text_pack["attention_mask"]
        src_padding_mask = ~(src_mask.bool())

        src_embed = self._add_positional_encoding(self.embedding(src_ids))
        semantic_seq = self.semantic_encoder(src_embed, src_key_padding_mask=src_padding_mask, snr=0.0)

        denom = src_mask.sum(dim=1, keepdim=True).clamp(min=1)
        masked_x = semantic_seq * src_mask.unsqueeze(-1)
        med_feature = masked_x.sum(dim=1) / denom

        self.med.add_to_stm(
            {
                "image_id": str(image_id),
                "caption_text": str(source_text),
                "token_ids": src_ids[0],
                "semantic_feature": med_feature[0],
                "dataset_id": str(dataset_id),
            }
        )
        return self.med.maybe_update()

    def _build_semantic_sequence(
        self,
        source_text: str | List[str],
        snr_db: float,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(source_text, str):
            texts = [source_text]
        elif isinstance(source_text, list):
            if len(source_text) == 0:
                raise RuntimeError("source_text list cannot be empty")
            texts = [str(x) for x in source_text]
        else:
            raise RuntimeError(f"Unsupported source_text type: {type(source_text)}")

        text_pack = self._encode_text(texts, device=device)
        src_ids = text_pack["token_ids"]
        src_mask = text_pack["attention_mask"]
        src_padding_mask = ~(src_mask.bool())

        src_embed = self._add_positional_encoding(self.embedding(src_ids))
        semantic_seq = self.semantic_encoder(src_embed, src_key_padding_mask=src_padding_mask, snr=snr_db)
        self._assert_shape_dtype("semantic_seq", semantic_seq, (src_ids.size(0), self.max_text_len, self.feature_dim), src_embed.dtype)
        return {
            "src_ids": src_ids,
            "src_mask": src_mask,
            "src_padding_mask": src_padding_mask,
            "semantic_seq": semantic_seq,
        }

    def _transmit_sequence(self, semantic_seq: torch.Tensor, snr_db: float) -> Dict[str, torch.Tensor]:
        channel_symbols = self.channel_encoder(semantic_seq, snr=snr_db)
        self._assert_shape_dtype(
            "channel_symbols",
            channel_symbols,
            (semantic_seq.size(0), self.max_text_len, self.channel_dim),
            semantic_seq.dtype,
        )
        received_symbols = self.channel(channel_symbols, snr_db=snr_db, normalize_power=True)
        self._assert_shape_dtype(
            "received_symbols",
            received_symbols,
            (semantic_seq.size(0), self.max_text_len, self.channel_dim),
            semantic_seq.dtype,
        )
        recovered_sequence = self.channel_decoder(received_symbols, snr=snr_db)
        self._assert_shape_dtype(
            "recovered_sequence",
            recovered_sequence,
            (semantic_seq.size(0), self.max_text_len, self.feature_dim),
            semantic_seq.dtype,
        )
        return {
            "channel_symbols": channel_symbols,
            "received_symbols": received_symbols,
            "recovered_sequence": recovered_sequence,
        }

    def _decode_with_teacher(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        target_ids: torch.Tensor,
        snr_db: float,
        device: torch.device,
    ) -> Dict[str, torch.Tensor | bool]:
        decoder_input_ids = self._prepare_decoder_inputs(target_ids)
        tgt_embed = self._add_positional_encoding(self.embedding(decoder_input_ids))
        tgt_mask = (decoder_input_ids != self.tokenizer.pad_id).long()
        tgt_padding_mask = ~(tgt_mask.bool())
        causal_mask = self._build_causal_mask(decoder_input_ids.size(1), device=device)
        dec_out = self.semantic_decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            snr=snr_db,
        )
        return {
            "logits": self.lm_head(dec_out),
            "used_shift_right": True,
            "used_causal_mask": True,
        }

    def forward_channel_phase(
        self,
        snr_db: float,
        image: Image.Image | torch.Tensor | None = None,
        source_text: Optional[str | List[str]] = None,
    ) -> Dict[str, Any]:
        device = next(self.parameters()).device
        if source_text is None:
            if image is None:
                raise RuntimeError("forward_channel_phase requires either image or source_text")
            source_text = self.sender_ckb.forward(image)

        semantic_pack = self._build_semantic_sequence(source_text=source_text, snr_db=snr_db, device=device)
        # Channel phase 将 semantic_seq detach，防止梯度流回 semantic encoder/NAM
        tx_pack = self._transmit_sequence(semantic_seq=semantic_pack["semantic_seq"].detach(), snr_db=snr_db)
        return {
            "source_text": source_text,
            "semantic_seq": semantic_pack["semantic_seq"],
            "semantic_seq_detached": semantic_pack["semantic_seq"].detach(),
            "semantic_seq_teacher": semantic_pack["semantic_seq"].detach(),
            "channel_symbols": tx_pack["channel_symbols"],
            "received_symbols": tx_pack["received_symbols"],
            "recovered_sequence": tx_pack["recovered_sequence"],
            "recovered_seq": tx_pack["recovered_sequence"],
            "source_token_ids": semantic_pack["src_ids"],
            "source_attention_mask": semantic_pack["src_mask"],
            "padding_mask": semantic_pack["src_padding_mask"],
        }

    def forward_semantic_phase(
        self,
        snr_db: float,
        image: Image.Image | torch.Tensor | None = None,
        tgt_ids: Optional[torch.Tensor] = None,
        source_text: Optional[str | List[str]] = None,
    ) -> Dict[str, Any]:
        device = next(self.parameters()).device
        channel_phase = self.forward_channel_phase(image=image, snr_db=snr_db, source_text=source_text)
        src_ids: torch.Tensor = channel_phase["source_token_ids"]
        src_padding_mask = ~(channel_phase["source_attention_mask"].bool())
        if tgt_ids is None:
            tgt_ids = src_ids
        tgt_ids = tgt_ids.to(device)
        decode_pack = self._decode_with_teacher(
            memory=channel_phase["recovered_sequence"],
            memory_key_padding_mask=src_padding_mask,
            target_ids=tgt_ids,
            snr_db=snr_db,
            device=device,
        )
        channel_phase["logits"] = decode_pack["logits"]
        channel_phase["target_ids"] = tgt_ids
        channel_phase["used_shift_right"] = bool(decode_pack["used_shift_right"])
        channel_phase["used_causal_mask"] = bool(decode_pack["used_causal_mask"])
        return channel_phase

    def forward_joint_phase(
        self,
        snr_db: float,
        image: Image.Image | torch.Tensor | None = None,
        tgt_ids: Optional[torch.Tensor] = None,
        source_text: Optional[str | List[str]] = None,
        image_id: str = "sample_0",
        dataset_id: str = "default",
    ) -> Dict[str, Any]:
        out = self.forward_semantic_phase(
            image=image,
            snr_db=snr_db,
            tgt_ids=tgt_ids,
            source_text=source_text,
        )
        out["med_status"] = self.get_med_state()
        return out

    def get_med_state(self) -> Dict[str, int]:
        if self.med is None:
            return {"enabled": 0, "stm_size": 0, "ltm_size": 0}
        return {"enabled": 1, "stm_size": len(self.med.stm), "ltm_size": len(self.med.ltm)}

    def sample_med_batch(self, batch_size: int, stm_ratio: float = 0.5):
        if self.med is None:
            raise RuntimeError("MED is disabled; sample_med_batch is not allowed.")
        return self.med.sample_train_batch(batch_size=batch_size, stm_ratio=stm_ratio)

    def forward_text_train(
        self,
        image: Image.Image | torch.Tensor,
        snr_db: float,
        tgt_ids: Optional[torch.Tensor] = None,
        source_text: Optional[str] = None,
        image_id: str = "sample_0",
        dataset_id: str = "default",
    ) -> Dict[str, torch.Tensor | str]:
        out = self.forward_joint_phase(
            image=image,
            snr_db=snr_db,
            tgt_ids=tgt_ids,
            source_text=source_text,
            image_id=image_id,
            dataset_id=dataset_id,
        )
        return {
            "source_text": out["source_text"],
            "logits": out["logits"],
            "target_ids": out["target_ids"],
            "med_status": out["med_status"],
        }

    def _beam_search_decode(
        self,
        memory: torch.Tensor,
        src_padding_mask: torch.Tensor,
        snr_db: float,
        beam_size: int = 4,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Beam search 解码（仅支持 B=1 推理）。

        相比贪心解码的优势：
        - 保留多条候选路径，避免早期错误 token 造成后续雪崩
        - 以 length-normalized 分数选择最优完整序列

        Args:
            memory:           channel decoder 输出 [1, L, D]
            src_padding_mask: [1, L] True 表示 padding
            snr_db:           当前信噪比
            beam_size:        beam 宽度（默认 4）
            device:           目标设备

        Returns:
            recovered_ids: [1, seq_len]
        """
        bos_id  = int(self.tokenizer.bos_id)
        eos_id  = int(self.tokenizer.eos_id)
        pad_id  = int(self.tokenizer.pad_id)
        min_len = min(4, max(2, self.max_text_len))
        if device is None:
            device = next(self.parameters()).device

        # 将 memory 从 [1, L, D] 扩展为 [beam, L, D]
        exp_memory = memory.expand(beam_size, -1, -1).contiguous()
        exp_mask   = src_padding_mask.expand(beam_size, -1).contiguous()

        # seqs:   [beam, seq_len]  当前活跃假设
        # scores: [beam]           累计 log-prob
        seqs   = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=device)
        scores = torch.full((beam_size,), float("-inf"), device=device)
        scores[0] = 0.0  # 第一条 beam 开始时激活

        completed: list = []  # (length-normalized-score, tensor)

        for step in range(self.max_text_len - 1):
            if len(completed) >= beam_size:
                break
            n_active = seqs.size(0)
            if n_active == 0:
                break

            tgt_embed    = self._add_positional_encoding(self.embedding(seqs))
            tgt_pad_mask = ~(seqs != pad_id).bool()
            causal_mask  = self._build_causal_mask(seqs.size(1), device=device)
            dec_out = self.semantic_decoder(
                tgt=tgt_embed,
                memory=exp_memory[:n_active],
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=exp_mask[:n_active],
                snr=snr_db,
            )
            logits = self.lm_head(dec_out)[:, -1, :].clone()  # [n_active, V]
            logits[:, pad_id] = float("-inf")
            if step > 0:
                logits[:, bos_id] = float("-inf")
            if step < min_len - 1:
                logits[:, eos_id] = float("-inf")

            log_probs  = F.log_softmax(logits, dim=-1)             # [n_active, V]
            cand       = scores[:n_active].unsqueeze(1) + log_probs  # [n_active, V]
            cand_flat  = cand.view(-1)                               # [n_active*V]
            vocab_size = log_probs.size(-1)

            k_expand = min(beam_size * 2, cand_flat.size(0))
            top_scores_t, top_ids_t = cand_flat.topk(k_expand)

            new_seqs:   list = []
            new_scores: list = []

            for flat_id, sc in zip(top_ids_t.tolist(), top_scores_t.tolist()):
                beam_id  = flat_id // vocab_size
                token_id = flat_id %  vocab_size
                new_seq  = torch.cat([
                    seqs[beam_id],
                    torch.tensor([token_id], dtype=torch.long, device=device),
                ])
                if token_id == eos_id:
                    length_norm = new_seq.size(0) ** 0.6
                    completed.append((sc / length_norm, new_seq))
                else:
                    if len(new_seqs) < beam_size:
                        new_seqs.append(new_seq)
                        new_scores.append(sc)
                if len(new_seqs) >= beam_size and len(completed) >= beam_size:
                    break

            if not new_seqs:
                break

            # 将所有活跃假设 padding 到等长
            max_slen = max(s.size(0) for s in new_seqs)
            n_new    = len(new_seqs)
            seqs     = torch.full((n_new, max_slen), pad_id, dtype=torch.long, device=device)
            for i, s in enumerate(new_seqs):
                seqs[i, :s.size(0)] = s
            scores   = torch.tensor(new_scores, dtype=torch.float, device=device)

            # 同步扩展 memory / mask 至新 beam 数量
            exp_memory = memory.expand(n_new, -1, -1).contiguous()
            exp_mask   = src_padding_mask.expand(n_new, -1).contiguous()

        # 若没有 EOS 完成的序列，将剩余活跃假设作为后备
        if not completed:
            for i in range(seqs.size(0)):
                sc = scores[i].item()
                if sc != float("-inf"):
                    length_norm = max(seqs[i].size(0), 1) ** 0.6
                    completed.append((sc / length_norm, seqs[i]))
        if not completed:
            return torch.full((1, 1), bos_id, dtype=torch.long, device=device)

        best_seq = max(completed, key=lambda x: x[0])[1]
        return best_seq.unsqueeze(0)  # [1, seq_len]

    @torch.inference_mode()
    def infer_full(
        self,
        image: Image.Image | torch.Tensor,
        snr_db: float,
        sd_height: int = 512,
        sd_width: int = 512,
        sd_steps: int = 30,
        sd_guidance: float = 7.5,
        sd_seed: int = 42,
        return_debug: bool = True,
        decode_strategy: str = "beam",
        beam_size: int = 4,
    ) -> Dict[str, object]:
        """完整推理流水线：图像 → 信道 → 重建图像。

        Args:
            decode_strategy: "beam"（默认）或 "greedy"。
                beam search 保留多条候选路径，避免贪心解码的误差雪崩问题。
                仅 B=1 时支持 beam；B>1 自动回退到 greedy。
            beam_size: beam search 宽度（decode_strategy="beam" 时有效）。
        """
        device = next(self.parameters()).device
        source_text = self.sender_ckb.forward(image)
        channel_phase = self.forward_channel_phase(image=image, snr_db=snr_db, source_text=source_text)
        src_ids = channel_phase["source_token_ids"]
        src_padding_mask = ~(channel_phase["source_attention_mask"].bool())
        memory = channel_phase["recovered_sequence"]
        channel_symbols = channel_phase["channel_symbols"]
        received_symbols = channel_phase["received_symbols"]
        recovered_sequence = channel_phase["recovered_sequence"]

        B = src_ids.size(0)
        if decode_strategy == "beam" and B == 1:
            # ── Beam Search 解码（推荐）──────────────────────────────────────
            # 相比贪心解码：保留 beam_size 条候选路径，避免单步错误引发连锁错误
            recovered_ids = self._beam_search_decode(
                memory=memory,
                src_padding_mask=src_padding_mask,
                snr_db=snr_db,
                beam_size=beam_size,
                device=device,
            )
        else:
            # ── Greedy 解码（原始实现；B>1 时自动使用）───────────────────────
            generated_ids = torch.full(
                (B, 1),
                int(self.tokenizer.bos_id),
                dtype=torch.long,
                device=device,
            )
            eos_id = int(self.tokenizer.eos_id)
            min_generation_len = min(4, max(2, self.max_text_len))

            for _ in range(self.max_text_len - 1):
                tgt_embed = self._add_positional_encoding(self.embedding(generated_ids))
                tgt_pad_mask = ~(generated_ids != self.tokenizer.pad_id).bool()
                causal_mask = self._build_causal_mask(generated_ids.size(1), device=device)
                dec_out = self.semantic_decoder(
                    tgt=tgt_embed,
                    memory=memory,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                    memory_key_padding_mask=src_padding_mask,
                    snr=snr_db,
                )
                logits = self.lm_head(dec_out)
                logits = logits.clone()
                logits[:, -1, int(self.tokenizer.pad_id)] = float("-inf")
                # NOTE: BOS=CLS=101 — 训练时 position 0 的 target 就是 CLS(101)，
                # 如果在 position 0 也阻塞 BOS，会导致第一个 token 就偏移，
                # 后续自回归生成产生级联错误。仅在 position 1+ 阻塞 BOS。
                if generated_ids.size(1) > 1:
                    logits[:, -1, int(self.tokenizer.bos_id)] = float("-inf")
                if generated_ids.size(1) < min_generation_len:
                    logits[:, -1, eos_id] = float("-inf")
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_id], dim=1)

                if bool((next_id == eos_id).all()):
                    break

            recovered_ids = generated_ids

        recovered_text = self.tokenizer.decode(recovered_ids)[0]

        # SD receiver: 仅在 SD pipeline 已加载时执行图像重建
        if self.receiver_ckb.use_real_ckb:
            reconstructed_image = self.receiver_ckb.forward(
                recovered_text,
                height=sd_height,
                width=sd_width,
                num_inference_steps=sd_steps,
                guidance_scale=sd_guidance,
                seed=sd_seed,
            )
        else:
            reconstructed_image = None

        result = {
            "source_text": source_text,
            "recovered_text": recovered_text,
            "reconstructed_image": reconstructed_image,
            "token_ids": recovered_ids,
            "source_token_ids": src_ids,
            "generated_ids": recovered_ids,
            "channel_symbols": channel_symbols,
            "received_symbols": received_symbols,
            "recovered_sequence": recovered_sequence,
        }

        if not return_debug:
            return {
                "source_text": result["source_text"],
                "recovered_text": result["recovered_text"],
                "reconstructed_image": result["reconstructed_image"],
                "token_ids": result["token_ids"],
            }
        return result


def smoke_test() -> None:
    base_dir = Path(__file__).resolve().parent
    downloaded_models_dir = base_dir.parent / "data" / "assets" / "downloaded_models"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VLMCscSystem(
        feature_dim=128,
        max_text_len=24,
        channel_type="awgn",
        use_real_ckb=False,
        blip_dir=downloaded_models_dir / "blip",
        sd_dir=downloaded_models_dir / "sd15",
        device=device,
    ).to(device)

    image = Image.new("RGB", (224, 224), color=(120, 140, 160))
    train_out = model.forward_text_train(image=image, snr_db=5.0)
    infer_out = model.infer_full(image=image, snr_db=5.0)

    logits = train_out["logits"]
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("Smoke test failed: logits is not a tensor.")

    print("[SMOKE] forward_text_train logits shape:", tuple(logits.shape))
    print("[SMOKE] source_text:", infer_out["source_text"])
    print("[SMOKE] recovered_text:", infer_out["recovered_text"])
    print("[SMOKE] system pipeline run: OK")


if __name__ == "__main__":
    smoke_test()
