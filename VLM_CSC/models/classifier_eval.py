"""Image classifier evaluator for ST(.) and SSQ computation.

Input:
- images: Tensor[B,3,H,W]
Output:
- logits/probabilities for classification accuracy.

Paper traceability:
- ST(.) as classification accuracy is 论文明确写出.
- Choice of classifier network is 为复现做的合理实现选择.
"""

from torch import Tensor, nn
import torch

from eval.metrics_image import classification_accuracy


class ClassifierEvaluator(nn.Module):
    """Classifier used for SSQ evaluation."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images [B,3,H,W], got {tuple(images.shape)}")
        feat = self.backbone(images).flatten(1)
        return self.head(feat)

    def accuracy(self, images: Tensor, labels: Tensor) -> Tensor:
        logits = self.forward(images)
        acc = classification_accuracy(logits, labels)
        return torch.tensor(acc, dtype=torch.float32)
