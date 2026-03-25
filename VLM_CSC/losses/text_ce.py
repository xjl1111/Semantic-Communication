"""Text-level cross entropy loss.

Input:
- logits: Tensor[B,T,V]
- targets: Tensor[B,T]
- attention_mask: Tensor[B,T]

Output:
- scalar CE loss

Paper traceability:
- Text consistency CE is 论文明确写出.
"""

from torch import Tensor
import torch.nn.functional as F


def text_cross_entropy(logits: Tensor, targets: Tensor, attention_mask: Tensor) -> Tensor:
    """Compute masked CE loss for token prediction."""
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B,T,V], got {tuple(logits.shape)}")
    if targets.ndim != 2:
        raise ValueError(f"Expected targets [B,T], got {tuple(targets.shape)}")
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected attention_mask [B,T], got {tuple(attention_mask.shape)}")

    bsz, seq_len, vocab_size = logits.shape
    if targets.shape != (bsz, seq_len) or attention_mask.shape != (bsz, seq_len):
        raise ValueError("targets/attention_mask shape mismatch with logits")

    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    flat_mask = attention_mask.reshape(-1).float()

    per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    masked = per_token * flat_mask
    denom = flat_mask.sum().clamp_min(1.0)
    return masked.sum() / denom
