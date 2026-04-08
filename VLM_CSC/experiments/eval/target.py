from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image


@dataclass
class SSQResult:
    ssq: float
    st_original: float
    st_reconstructed: float


class CatsDogsDownstreamClassifier:
    """Strict downstream classifier for Cats-vs-Dogs used by Fig.7 SSQ.

    按优先级依次尝试以下 CLIP 变体（均需本地缓存）：
      1. openai/clip-vit-large-patch14  (ViT-L/14, ~430M, 最佳精度)
      2. openai/clip-vit-base-patch32   (ViT-B/32, ~150M, 降级方案)

    backend 统一报告为 "clip_zeroshot"，可通过 clip_model_variant
    属性（"vitl14" / "vitb32"）查看实际加载的变体。
    严格模式下若两者均不可用则直接抛出异常。
    """

    # 按优先级排列：最强模型在前
    _CLIP_CANDIDATES = [
        ("openai/clip-vit-large-patch14", "vitl14"),
        ("openai/clip-vit-base-patch32",  "vitb32"),
    ]

    def __init__(self, device: str = "cuda", finetuned_clip_path: str = ""):
        self.device = device
        self.backend = ""
        self.clip_model_variant = ""          # "vitl14" | "vitb32"
        self.degradation_notes: List[str] = []

        self.clip_model = None
        self.clip_processor = None
        self.clip_text_features = None
        self.clip_head = None                 # Linear head for finetuned mode

        self.resnet_model = None
        self.resnet_preprocess = None

        self._try_init_clip()
        if self.backend == "":
            raise RuntimeError("Strict protocol requires local CLIP cache; no alternate backend is allowed.")

        # 如果提供了微调权重路径且文件存在，切换到 finetuned 模式
        if finetuned_clip_path and Path(finetuned_clip_path).exists():
            self._load_finetuned_head(finetuned_clip_path)

    def _try_init_clip(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        last_exc: Exception | None = None
        for model_name, variant in self._CLIP_CANDIDATES:
            try:
                processor = CLIPProcessor.from_pretrained(
                    model_name,
                    local_files_only=True,
                    use_fast=True,
                )
                model = CLIPModel.from_pretrained(
                    model_name,
                    local_files_only=True,
                ).to(self.device)
                model.eval()

                texts = ["a photo of a cat", "a photo of a dog"]
                text_inputs = processor(text=texts, return_tensors="pt", padding=True)
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                with torch.no_grad():
                    text_features = model.get_text_features(**text_inputs)
                    text_features = torch.nn.functional.normalize(text_features, dim=-1)

                self.clip_processor = processor
                self.clip_model = model
                self.clip_text_features = text_features
                self.backend = "clip_zeroshot"
                self.clip_model_variant = variant
                print(f"[CatsDogsClassifier] 已加载 CLIP 分类器: {model_name} ({variant})")
                return
            except Exception as exc:
                print(
                    f"[CatsDogsClassifier] {model_name} 本地缓存不可用，尝试下一个 "
                    f"({type(exc).__name__}: {exc})"
                )
                last_exc = exc

        raise RuntimeError(
            f"所有 CLIP 变体均不在本地缓存中，无法初始化分类器。"
            f"请先运行 vlm_csc.assets.download_clip 或手动下载 clip-vit-large-patch14。"
            f"最后一个错误: {last_exc}"
        ) from last_exc

    def _load_finetuned_head(self, path: str) -> None:
        """Load a trained Linear(feat_dim, 2) classification head."""
        import torch.nn as nn
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        feat_dim = ckpt["head.weight"].shape[1]  # e.g. 768 for ViT-L/14
        head = nn.Linear(feat_dim, 2).to(self.device)
        head.load_state_dict({k.replace("head.", ""): v for k, v in ckpt.items() if k.startswith("head.")})
        head.eval()
        self.clip_head = head
        self.backend = "clip_finetuned"
        print(f"[CatsDogsClassifier] 已加载微调 CLIP 分类头: {path}")

    @torch.no_grad()
    def predict_label(self, image: Image.Image) -> int:
        image = image.convert("RGB")

        if self.backend in ("clip_zeroshot", "clip_finetuned"):
            image_inputs = self.clip_processor(images=image, return_tensors="pt")
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
            image_features = self.clip_model.get_image_features(**image_inputs)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)

            if self.backend == "clip_finetuned" and self.clip_head is not None:
                logits = self.clip_head(image_features)
            else:
                logits = image_features @ self.clip_text_features.T

            pred = int(logits.argmax(dim=-1).item())
            return pred

        raise RuntimeError("Unknown downstream classifier backend.")


def compute_accuracy(preds: List[int], labels: List[int]) -> float:
    if len(labels) == 0:
        return 0.0
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    return float(correct / len(labels))


def compute_classification_accuracy(preds: List[int], labels: List[int]) -> float:
    return compute_accuracy(preds, labels)


def compute_ssq_from_accuracies(st_original: float, st_reconstructed: float) -> float:
    if st_original <= 1e-12:
        raise RuntimeError("st_original is zero; SSQ undefined under strict protocol.")
    return float(st_reconstructed / st_original)

def compute_ssq(preds_original: List[int], preds_reconstructed: List[int], labels: List[int]) -> SSQResult:
    st_original = compute_accuracy(preds_original, labels)
    st_reconstructed = compute_accuracy(preds_reconstructed, labels)
    ssq = compute_ssq_from_accuracies(st_original=st_original, st_reconstructed=st_reconstructed)
    return SSQResult(ssq=ssq, st_original=st_original, st_reconstructed=st_reconstructed)


def _tokenize_text(text: str) -> List[str]:
    return str(text).strip().lower().split()


def compute_bleu_n(references: List[str], hypotheses: List[str], n: int) -> float:
    if len(references) == 0 or len(hypotheses) == 0:
        raise RuntimeError("BLEU requires non-empty references and hypotheses under strict protocol.")
    if len(references) != len(hypotheses):
        raise RuntimeError("BLEU input size mismatch between references and hypotheses.")
    if n not in {1, 2}:
        raise RuntimeError(f"Unsupported BLEU order: {n}")

    bleu_module = importlib.import_module("nltk.translate.bleu_score")
    SmoothingFunction = getattr(bleu_module, "SmoothingFunction")
    corpus_bleu = getattr(bleu_module, "corpus_bleu")

    refs = [[_tokenize_text(ref)] for ref in references]
    hyps = [_tokenize_text(hyp) for hyp in hypotheses]
    smoothing = SmoothingFunction().method1
    weights = (1.0, 0.0, 0.0, 0.0) if n == 1 else (0.5, 0.5, 0.0, 0.0)
    return float(corpus_bleu(refs, hyps, weights=weights, smoothing_function=smoothing))


def compute_bleu1(references: List[str], hypotheses: List[str]) -> float:
    return compute_bleu_n(references, hypotheses, n=1)


def compute_bleu2(references: List[str], hypotheses: List[str]) -> float:
    return compute_bleu_n(references, hypotheses, n=2)


def compute_bleu(references: List[str], hypotheses: List[str]) -> float:
    return compute_bleu1(references, hypotheses)


def compute_compression_ratio(source_token_lengths: List[int], transmitted_token_lengths: List[int]) -> float:
    if len(source_token_lengths) == 0 or len(transmitted_token_lengths) == 0:
        raise RuntimeError("Compression ratio requires non-empty lengths under strict protocol.")
    if len(source_token_lengths) != len(transmitted_token_lengths):
        raise RuntimeError("Compression ratio length mismatch.")
    src_total = sum(int(v) for v in source_token_lengths)
    tx_total = sum(int(v) for v in transmitted_token_lengths)
    if tx_total <= 0:
        raise RuntimeError("Transmitted length total must be > 0.")
    return float(src_total / tx_total)


def count_trainable_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def compute_lpips(original_images: List[Image.Image], reconstructed_images: List[Image.Image], device: str = "cuda") -> float:
    if len(original_images) == 0 or len(reconstructed_images) == 0:
        return 0.0
    if len(original_images) != len(reconstructed_images):
        raise RuntimeError("LPIPS input size mismatch between original and reconstructed images.")

    lpips = importlib.import_module("lpips")

    model = lpips.LPIPS(net="alex").to(device).eval()
    values = []
    with torch.no_grad():
        for img_a, img_b in zip(original_images, reconstructed_images):
            tensor_a = _pil_to_lpips_tensor(img_a, device)
            tensor_b = _pil_to_lpips_tensor(img_b, device)
            values.append(float(model(tensor_a, tensor_b).item()))
    return float(sum(values) / max(len(values), 1))


def _pil_to_lpips_tensor(image: Image.Image, device: str) -> torch.Tensor:
    x = torch.from_numpy(np.array(image.convert("RGB"), dtype="float32")).permute(2, 0, 1)
    x = (x / 127.5) - 1.0
    return x.unsqueeze(0).to(device)
