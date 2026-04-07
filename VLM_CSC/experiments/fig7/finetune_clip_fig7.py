"""CLIP 分类器微调脚本（Linear Probe on Frozen CLIP）

在 CatsVsDogs 训练集上训练一个 Linear(feat_dim, 2) 分类头，
CLIP 视觉编码器保持冻结。

用法（独立运行）：
    python finetune_clip_fig7.py

    或由 run_fig7.py --finetune_clip 自动调用。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from PIL import Image
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_EXP_DIR = _THIS_DIR.parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))


# ═══════════════════════════════════════════════════════════════════════════
#  数据集
# ═══════════════════════════════════════════════════════════════════════════

class CatDogImageDataset(Dataset):
    """从 train/cat/ 和 train/dog/ 加载图像，返回 (PIL.Image, label)。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.samples: list[tuple[Path, int]] = []
        for cls_name, label in [("cat", 0), ("dog", 1)]:
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                raise FileNotFoundError(f"找不到类别目录: {cls_dir}")
            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((img_path, label))
        print(f"[CLIP-FT] 数据集: {len(self.samples)} 张图像 ({self.root})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return image, label


# ═══════════════════════════════════════════════════════════════════════════
#  训练逻辑
# ═══════════════════════════════════════════════════════════════════════════

def _collate_fn(batch, processor, device):
    """自定义 collate: 用 CLIP processor 处理图像。"""
    images, labels = zip(*batch)
    inputs = processor(images=list(images), return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    return inputs, labels_t


def main(
    data_root: str | None = None,
    output_path: str | None = None,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> Path:
    """训练 CLIP Linear Probe 分类器。

    参数可通过函数调用传入（由 run_fig7.py 使用），
    也可通过命令行运行时使用 argparse 解析。
    """
    # ── 命令行参数（独立运行时使用）────────────────────────────────────────
    if data_root is None:
        parser = argparse.ArgumentParser(description="Finetune CLIP classifier (linear probe)")
        parser.add_argument("--data_root", type=str, default=None,
                            help="CatsVsDogs 训练集根目录 (含 cat/, dog/ 子目录)")
        parser.add_argument("--output", type=str, default=None,
                            help="保存分类头权重的路径")
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--batch_size", type=int, default=32)
        cli_args = parser.parse_args()
        data_root = cli_args.data_root
        output_path = cli_args.output
        epochs = cli_args.epochs
        lr = cli_args.lr
        batch_size = cli_args.batch_size

    # ── 默认路径 ──────────────────────────────────────────────────────────
    _root = Path(__file__).resolve().parents[3]
    if data_root is None:
        data_root = str(_root / "data" / "datasets" / "catsvsdogs" / "train")
    if output_path is None:
        output_path = str(
            _root / "VLM_CSC" / "data" / "experiments" / "fig7" / "finetuned_clip" / "clip_classifier.pth"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CLIP-FT] device={device}, epochs={epochs}, lr={lr}, batch_size={batch_size}")

    # ── 加载 CLIP ─────────────────────────────────────────────────────────
    from transformers import CLIPModel, CLIPProcessor

    _CLIP_CANDIDATES = [
        "openai/clip-vit-large-patch14",
        "openai/clip-vit-base-patch32",
    ]
    clip_model = None
    processor = None
    for model_name in _CLIP_CANDIDATES:
        try:
            processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True, use_fast=True)
            clip_model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device)
            clip_model.eval()
            print(f"[CLIP-FT] 已加载 CLIP: {model_name}")
            break
        except Exception as e:
            print(f"[CLIP-FT] {model_name} 不可用: {e}")
    if clip_model is None:
        raise RuntimeError("无法加载任何 CLIP 模型")

    # 冻结 CLIP
    for param in clip_model.parameters():
        param.requires_grad = False

    # 获取特征维度
    feat_dim = clip_model.config.projection_dim  # 768 for ViT-L/14, 512 for ViT-B/32
    print(f"[CLIP-FT] 特征维度: {feat_dim}")

    # ── 构建数据集 ────────────────────────────────────────────────────────
    full_dataset = CatDogImageDataset(data_root)
    val_size = max(1, int(len(full_dataset) * 0.1))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"[CLIP-FT] train={train_size}, val={val_size}")

    # ── 提取特征（一次性，节省重复前向传播）────────────────────────────────
    print("[CLIP-FT] 提取 CLIP 图像特征 ...")

    def extract_features(dataset, desc="extract"):
        # PIL Image 无法被 default_collate 处理，手动 batch 提取
        all_feats, all_labels = [], []
        indices = list(range(len(dataset)))
        for start in tqdm(range(0, len(indices), batch_size), desc=desc, leave=False):
            batch_idx = indices[start : start + batch_size]
            batch_images = []
            batch_labels = []
            for i in batch_idx:
                img, lbl = dataset[i]
                batch_images.append(img)
                batch_labels.append(lbl)
            inputs = processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                features = clip_model.get_image_features(**inputs)
                features = torch.nn.functional.normalize(features, dim=-1)
            all_feats.append(features.cpu())
            all_labels.extend(batch_labels)
        return torch.cat(all_feats, dim=0), torch.tensor(all_labels, dtype=torch.long)

    train_feats, train_labels = extract_features(train_ds, "train features")
    val_feats, val_labels = extract_features(val_ds, "val features")
    print(f"[CLIP-FT] 特征提取完成: train={train_feats.shape}, val={val_feats.shape}")

    # ── 训练 Linear Head ─────────────────────────────────────────────────
    head = nn.Linear(feat_dim, 2).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_feats_d = train_feats.to(device)
    train_labels_d = train_labels.to(device)
    val_feats_d = val_feats.to(device)
    val_labels_d = val_labels.to(device)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        head.train()
        # mini-batch SGD on features
        perm = torch.randperm(train_feats_d.size(0), device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, train_feats_d.size(0), batch_size):
            idx = perm[start : start + batch_size]
            logits = head(train_feats_d[idx])
            loss = criterion(logits, train_labels_d[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # 验证
        head.eval()
        with torch.no_grad():
            val_logits = head(val_feats_d)
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds == val_labels_d).float().mean().item()

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"[CLIP-FT] epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {f"head.{k}": v.cpu().clone() for k, v in head.state_dict().items()}

    # ── 保存 ─────────────────────────────────────────────────────────────
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_p)
    print(f"[CLIP-FT] 最佳 val_acc={best_val_acc:.4f}, 保存至: {output_p}")
    return output_p


if __name__ == "__main__":
    main()
