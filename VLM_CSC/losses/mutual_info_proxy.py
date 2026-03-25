"""Mutual information proxy loss for channel training.

Paper traceability:
- Paper mentions minimizing mutual information objective conceptually.
- Concrete computable form is 为复现做的合理实现选择.
"""

from torch import Tensor
import torch
import torch.nn.functional as F


def channel_proxy_loss(decoded_feat: Tensor, original_feat: Tensor, encoded_symbols: Tensor, beta: float = 1e-3) -> Tensor:
    """Default proxy: MSE(decoded, original) + beta*power_penalty."""
    if decoded_feat.shape != original_feat.shape:
        raise ValueError(
            f"decoded/original shape mismatch: {tuple(decoded_feat.shape)} vs {tuple(original_feat.shape)}"
        )
    mse = F.mse_loss(decoded_feat, original_feat)
    symbol_power = torch.mean(encoded_symbols.pow(2), dim=-1)
    power_penalty = torch.mean((symbol_power - 1.0).pow(2))
    return mse + beta * power_penalty
