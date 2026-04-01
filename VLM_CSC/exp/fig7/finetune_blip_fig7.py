"""
finetune_blip_fig7.py — 在 Fig.7 CatsVsDogs 数据集上微调 BLIP ViT

┌─────────────────────────────────────────────────────────────────────┐
│  目的                                                               │
├─────────────────────────────────────────────────────────────────────┤
│  BLIP 原始预训练权重基于自然高清图像（COCO/CC3M），而 VLM-CSC 系统  │
│  的输入是 32×32 低分辨率图像（即使经 SR 上采样到 256×256 也仍有退   │
│  化伪影）。ViT 对这类退化图像的特征提取质量下降，导致生成的 caption  │
│  中类别词缺失或错误，限制了最终 SSQ。                               │
│                                                                     │
│  此脚本对 BLIP 的视觉编码器（ViT）做 image-captioning 微调，使其     │
│  更适应低分辨率退化图像，从而提升 caption 准确率（Level-A）→ SSQ。   │
└─────────────────────────────────────────────────────────────────────┘

策略：
  - 图像经过与推理时完全相同的 SR 上采样（LANCZOS + UnsharpMask）
  - 目标 caption = 类别相关的多样化模板（"a photo of a cat" 等）
  - 仅解冻 ViT 部分，文本解码器默认冻结（可通过 --unfreeze 控制）
  - 保存完整 BLIP 模型（HuggingFace 格式），可被 SenderCKB_BLIP 直接加载

用法：
    python finetune_blip_fig7.py                           # 使用 fig7_config 默认参数
    python finetune_blip_fig7.py --epochs 10               # 覆盖 epoch 数
    python finetune_blip_fig7.py --unfreeze all            # 解冻整个 BLIP
    python finetune_blip_fig7.py --max_per_class 50        # 调试：每类仅 50 张
    python finetune_blip_fig7.py --dry_run                 # 仅打印配置，不训练

训练完成后：
    1. 在 fig7_config.py 中设置 USE_FINETUNED_BLIP = True
    2. 运行  python run_fig7.py --mode train   重新训练 CSC 模型
    3. 运行  python run_fig7.py --mode eval    评估 SSQ

    或一步到位：
    python run_fig7.py --mode all --use_finetuned_blip
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image, ImageFilter
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_EXP_DIR = _THIS_DIR.parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from fig7_config import build_fig7_config


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SR 上采样（与 ImageUpsampler 保持完全一致）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sr_upsample(
    pil_image: Image.Image,
    target_size: int = 256,
    sharpen_radius: float = 2.0,
    sharpen_percent: int = 150,
    sharpen_threshold: int = 3,
) -> Image.Image:
    """复制 ImageUpsampler 的逻辑，确保微调预处理与推理一致。"""
    w, h = pil_image.size
    if w >= target_size and h >= target_size:
        return pil_image
    scale = max(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    upscaled = pil_image.resize((new_w, new_h), Image.LANCZOS)
    sharpened = upscaled.filter(
        ImageFilter.UnsharpMask(
            radius=sharpen_radius, percent=sharpen_percent, threshold=sharpen_threshold,
        )
    )
    return sharpened


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  数据集
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 每类多个表述模板，防止模型过拟合于单一句式
_CAPTION_TEMPLATES: Dict[str, List[str]] = {
    "cat": [
        "a photo of a cat",
        "a cat",
        "a cute cat",
        "a photo of an animal, a cat",
        "this is a cat",
        "an image of a cat",
        "a small cat",
    ],
    "dog": [
        "a photo of a dog",
        "a dog",
        "a cute dog",
        "a photo of an animal, a dog",
        "this is a dog",
        "an image of a dog",
        "a small dog",
    ],
}


class CatsDogsCaptionDataset(Dataset):
    """CatsVsDogs 图像-描述对数据集，用于 BLIP captioning 微调。"""

    def __init__(
        self,
        root_dir: str | Path,
        processor,
        apply_sr: bool = True,
        max_per_class: int = -1,
    ):
        self.processor = processor
        self.apply_sr = apply_sr
        self.samples: List[Tuple[str, str]] = []  # (image_path, class_name)

        root = Path(root_dir)
        for class_name in sorted(_CAPTION_TEMPLATES.keys()):
            class_dir = root / class_name
            if not class_dir.exists():
                print(f"[WARN] 类别目录不存在: {class_dir}")
                continue
            paths = sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
            )
            if max_per_class > 0:
                paths = paths[:max_per_class]
            for p in paths:
                self.samples.append((str(p), class_name))

        print(f"[DATASET] 加载 {len(self.samples)} 张图像 ({root_dir})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, class_name = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.apply_sr:
            img = sr_upsample(img)

        # 随机选择一个 caption 模板
        caption = random.choice(_CAPTION_TEMPLATES[class_name])

        # 使用 processor 编码
        inputs = self.processor(
            images=img,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )

        # squeeze batch dim, 将 input_ids 作为 labels
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = item["input_ids"].clone()
        return item


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  模型工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def unfreeze_blip_layers(model, strategy: str = "vit") -> Dict[str, int]:
    """设置 BLIP 各部分的可训练状态。

    strategy:
        "vit"      — 仅解冻 vision_model（ViT encoder）
        "vit_proj" — 解冻 vision_model + visual_projection
        "all"      — 解冻整个 BLIP（ViT + TextDecoder）
    """
    # 先冻结所有
    for param in model.parameters():
        param.requires_grad = False

    unfrozen, frozen = 0, 0

    if strategy == "vit":
        for name, param in model.named_parameters():
            if "vision_model" in name:
                param.requires_grad = True
                unfrozen += 1
            else:
                frozen += 1
    elif strategy == "vit_proj":
        for name, param in model.named_parameters():
            if "vision_model" in name or "visual_projection" in name:
                param.requires_grad = True
                unfrozen += 1
            else:
                frozen += 1
    elif strategy == "all":
        for param in model.parameters():
            param.requires_grad = True
            unfrozen += 1
    else:
        raise ValueError(f"Unknown unfreeze strategy: {strategy}")

    return {"unfrozen": unfrozen, "frozen": frozen, "strategy": strategy}


def get_lr_scheduler(optimizer, num_training_steps: int, warmup_ratio: float = 0.1):
    """Linear warmup + cosine decay scheduler."""
    warmup_steps = int(num_training_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(num_training_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  训练 / 验证
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_one_epoch(
    model, dataloader, optimizer, scheduler, device, epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [train]", leave=False)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, dataloader, device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="  [val]", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_caption_accuracy(
    model, processor, dataloader, device, max_batches: int = 20,
) -> Dict[str, float]:
    """在验证集上生成 caption，统计类别词准确率。"""
    model.eval()
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(
        tqdm(dataloader, desc="  [accuracy]", leave=False),
    ):
        if batch_idx >= max_batches:
            break
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]

        # 生成 caption
        generated_ids = model.generate(pixel_values=pixel_values, max_new_tokens=32)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # 解码 labels 以获取真实类别
        label_texts = processor.batch_decode(labels, skip_special_tokens=True)

        for gen_text, label_text in zip(generated_texts, label_texts):
            gen_lower = gen_text.lower()
            label_lower = label_text.lower()
            # 从 label 推断真实类别
            if "cat" in label_lower:
                gt_class = "cat"
            elif "dog" in label_lower:
                gt_class = "dog"
            else:
                continue
            if gt_class in gen_lower:
                correct += 1
            total += 1

    accuracy = correct / max(total, 1) * 100
    return {"accuracy": round(accuracy, 2), "correct": correct, "total": total}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  主函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="在 CatsVsDogs 上微调 BLIP ViT（提升低分辨率图像的 caption 准确率）",
    )
    parser.add_argument("--epochs",        type=int,   default=None, help="覆盖 VIT_FT_EPOCHS")
    parser.add_argument("--lr",            type=float, default=None, help="覆盖 VIT_FT_LR")
    parser.add_argument("--batch_size",    type=int,   default=None, help="覆盖 VIT_FT_BATCH_SIZE")
    parser.add_argument("--unfreeze",      type=str,   default=None,
                        choices=["vit", "vit_proj", "all"],
                        help="覆盖 VIT_FT_UNFREEZE")
    parser.add_argument("--max_per_class", type=int,   default=-1,
                        help="每类最大样本数（-1=全部，调试时设为 50）")
    parser.add_argument("--val_ratio",     type=float, default=0.2,
                        help="验证集比例（默认 0.2）")
    parser.add_argument("--dry_run",       action="store_true",
                        help="仅打印配置，不训练")
    args = parser.parse_args()

    # ── 加载配置 ──────────────────────────────────────────────────────────────
    cfg = build_fig7_config()

    epochs       = args.epochs    or cfg.get("vit_ft_epochs", 5)
    lr           = args.lr        or cfg.get("vit_ft_lr", 2e-5)
    batch_size   = args.batch_size or cfg.get("vit_ft_batch_size", 8)
    unfreeze     = args.unfreeze  or cfg.get("vit_ft_unfreeze", "vit")
    warmup_ratio = cfg.get("vit_ft_warmup_ratio", 0.1)

    # 始终从原始 BLIP 目录开始微调（不从已有 finetuned 权重继续）
    from common.shared_config import build_shared_paths
    blip_dir   = build_shared_paths()["blip_ckb_dir"]
    output_dir = cfg["finetuned_blip_dir"]
    train_dir  = cfg["train_split_dir"]

    caption_mode = cfg.get("caption_mode", "sr_prompt")
    # SR 是否启用取决于 caption_mode
    _SR_MODES = {"sr", "sr_prompt", "blip2", "baseline"}
    apply_sr = caption_mode in _SR_MODES

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  BLIP ViT 微调 — Fig.7 CatsVsDogs")
    print("=" * 70)
    print(f"  原始 BLIP 目录:  {blip_dir}")
    print(f"  输出目录:        {output_dir}")
    print(f"  训练数据:        {train_dir}")
    print(f"  caption_mode:    {caption_mode} (SR={'ON' if apply_sr else 'OFF'})")
    print(f"  解冻策略:        {unfreeze}")
    print(f"  Epochs:          {epochs}")
    print(f"  LR:              {lr}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Warmup ratio:    {warmup_ratio}")
    print(f"  设备:            {device}")
    print(f"  max_per_class:   {args.max_per_class}")
    print("=" * 70)

    if args.dry_run:
        print("[DRY_RUN] 仅打印配置，不执行训练。")
        return

    # ── [1/5] 加载模型 ────────────────────────────────────────────────────────
    from transformers import BlipForConditionalGeneration, BlipProcessor

    print("\n[1/5] 加载原始 BLIP 模型...")
    processor = BlipProcessor.from_pretrained(
        blip_dir, use_fast=False, local_files_only=True,
    )
    model = BlipForConditionalGeneration.from_pretrained(
        blip_dir, local_files_only=True,
    )
    model.to(device)

    # ── [2/5] 解冻指定层 ──────────────────────────────────────────────────────
    print("[2/5] 设置可训练参数...")
    freeze_info = unfreeze_blip_layers(model, strategy=unfreeze)
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable:,} / {total_params:,} ({100*trainable/total_params:.1f}%)")
    print(f"  解冻策略:   {freeze_info['strategy']} "
          f"({freeze_info['unfrozen']} layers unfrozen, {freeze_info['frozen']} frozen)")

    # ── [3/5] 创建数据集 ──────────────────────────────────────────────────────
    print("[3/5] 创建数据集...")
    full_dataset = CatsDogsCaptionDataset(
        root_dir=train_dir,
        processor=processor,
        apply_sr=apply_sr,
        max_per_class=args.max_per_class,
    )

    # 按 val_ratio 划分训练/验证
    n_total = len(full_dataset)
    n_val   = max(int(n_total * args.val_ratio), 1)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  训练集: {n_train},  验证集: {n_val}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # ── 优化器 + LR 调度器 ────────────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    num_training_steps = len(train_loader) * epochs
    scheduler = get_lr_scheduler(optimizer, num_training_steps, warmup_ratio)

    # ── [4/5] 训练前评估（baseline）──────────────────────────────────────────
    print("[4/5] 训练前评估...")
    pre_val_loss = validate(model, val_loader, device)
    pre_accuracy = evaluate_caption_accuracy(model, processor, val_loader, device)
    print(f"  [Before] val_loss={pre_val_loss:.4f}, "
          f"caption_accuracy={pre_accuracy['accuracy']:.1f}% "
          f"({pre_accuracy['correct']}/{pre_accuracy['total']})")

    # ── [5/5] 训练循环 ───────────────────────────────────────────────────────
    print("[5/5] 开始训练...")
    best_val_loss = float("inf")
    best_epoch = -1
    history: List[Dict] = []

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
        )
        val_loss = validate(model, val_loader, device)
        elapsed = time.time() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            # 保存最佳模型（HuggingFace 格式）
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir, safe_serialization=True)
            processor.save_pretrained(output_dir)

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "is_best": is_best,
            "elapsed_s": round(elapsed, 1),
        })

        flag = " ★ best" if is_best else ""
        print(f"  Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"time={elapsed:.0f}s{flag}")

    # ── 训练后评估 ────────────────────────────────────────────────────────────
    model_best = BlipForConditionalGeneration.from_pretrained(
        output_dir, local_files_only=True,
    ).to(device)
    model_best.eval()
    post_val_loss = validate(model_best, val_loader, device)
    post_accuracy = evaluate_caption_accuracy(model_best, processor, val_loader, device)

    delta = post_accuracy["accuracy"] - pre_accuracy["accuracy"]

    print()
    print("=" * 70)
    print("  微调完成")
    print("=" * 70)
    print(f"  最佳 epoch:       {best_epoch + 1} (val_loss={best_val_loss:.4f})")
    print(f"  模型保存至:       {output_dir}")
    print()
    print(f"  ┌─ Before ──────────────────────────────────────────┐")
    print(f"  │  val_loss={pre_val_loss:.4f}  "
          f"caption_accuracy={pre_accuracy['accuracy']:.1f}%  │")
    print(f"  ├─ After ───────────────────────────────────────────┤")
    print(f"  │  val_loss={post_val_loss:.4f}  "
          f"caption_accuracy={post_accuracy['accuracy']:.1f}%  │")
    print(f"  ├─ Delta ───────────────────────────────────────────┤")
    print(f"  │  accuracy {delta:+.1f}%                                 │")
    print(f"  └───────────────────────────────────────────────────┘")
    print()
    print("  后续步骤:")
    print("    方法 A — 修改配置文件:")
    print("      1. 在 fig7_config.py 中设置 USE_FINETUNED_BLIP = True")
    print("      2. python run_fig7.py --mode all")
    print()
    print("    方法 B — 命令行一次性:")
    print("      python run_fig7.py --mode all --use_finetuned_blip")
    print("=" * 70)

    # ── 保存训练日志 ──────────────────────────────────────────────────────────
    log_path = Path(output_dir) / "finetune_log.json"
    log_data = {
        "config": {
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "unfreeze": unfreeze,
            "warmup_ratio": warmup_ratio,
            "caption_mode": caption_mode,
            "apply_sr": apply_sr,
            "blip_dir": blip_dir,
            "output_dir": output_dir,
            "max_per_class": args.max_per_class,
            "train_samples": n_train,
            "val_samples": n_val,
        },
        "results": {
            "pre_val_loss": round(pre_val_loss, 4),
            "pre_accuracy": pre_accuracy,
            "post_val_loss": round(post_val_loss, 4),
            "post_accuracy": post_accuracy,
            "delta_accuracy": round(delta, 2),
            "best_epoch": best_epoch + 1,
            "best_val_loss": round(best_val_loss, 4),
        },
        "history": history,
    }
    log_path.write_text(
        json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\n  训练日志: {log_path}")


if __name__ == "__main__":
    main()
