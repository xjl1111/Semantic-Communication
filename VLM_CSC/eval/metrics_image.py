"""Image metrics: ST(.) and SSQ utilities."""

from torch import Tensor


def classification_accuracy(logits: Tensor, labels: Tensor) -> float:
    """Compute classification accuracy."""
    if logits.ndim != 2:
        raise ValueError(f"Expected logits [B,C], got {tuple(logits.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"Expected labels [B], got {tuple(labels.shape)}")
    preds = logits.argmax(dim=1)
    correct = (preds == labels).float().mean().item()
    return float(correct)


def ssq(recon_acc: float, original_acc: float) -> float:
    """Compute SSQ = ST(S_hat)/ST(S)."""
    if original_acc <= 0:
        raise ValueError("original_acc must be > 0")
    return recon_acc / original_acc
