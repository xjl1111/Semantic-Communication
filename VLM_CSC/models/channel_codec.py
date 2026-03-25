"""Channel encoder/decoder with NAM interleaving.

Input/Output shapes:
- Input features: Tensor[B, 128] or Tensor[B, T, 128]
- Encoded symbols: Tensor[B, C]
- Decoded features: Tensor[B, 128]

Paper traceability:
- Hidden sizes 256/128 and symmetric decoder are 论文明确写出.
- Symbol dim C and power normalization details are 为复现做的合理实现选择.
"""

from typing import Optional

import torch
from torch import Tensor, nn

from .nam import NoiseAdaptiveModulator


class ChannelEncoder(nn.Module):
    """Encode semantic features into channel symbols."""

    def __init__(self, d_model: int = 128, hidden1: int = 256, hidden2: int = 128, symbol_dim: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.symbol_dim = symbol_dim
        self.fc1 = nn.Linear(d_model, hidden1)
        self.nam1 = NoiseAdaptiveModulator(d_model=hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.nam2 = NoiseAdaptiveModulator(d_model=hidden2)
        self.symbol_proj = nn.Linear(hidden2, symbol_dim)
        self.relu = nn.ReLU()

    def forward(self, semantic_features: Tensor, snr: Optional[Tensor] = None) -> Tensor:
        """Forward encode semantic features to channel symbols [B,C]."""
        if semantic_features.ndim == 3:
            if semantic_features.shape[-1] != self.d_model:
                raise ValueError(
                    f"Expected semantic_features shape [B,T,{self.d_model}], got {tuple(semantic_features.shape)}"
                )
            semantic_code = semantic_features.mean(dim=1)
        elif semantic_features.ndim == 2:
            if semantic_features.shape[-1] != self.d_model:
                raise ValueError(
                    f"Expected semantic_features shape [B,{self.d_model}], got {tuple(semantic_features.shape)}"
                )
            semantic_code = semantic_features
        else:
            raise ValueError(
                f"Expected semantic_features shape [B,{self.d_model}] or [B,T,{self.d_model}], got {tuple(semantic_features.shape)}"
            )

        batch_size = semantic_code.shape[0]
        if snr is None:
            snr = torch.zeros(batch_size, 1, device=semantic_code.device, dtype=semantic_code.dtype)
        else:
            snr = snr.to(device=semantic_code.device, dtype=semantic_code.dtype)

        x = self.fc1(semantic_code)
        x = self.nam1(snr, x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.nam2(snr, x)
        x = self.relu(x)

        symbols = self.symbol_proj(x)
        return power_normalize(symbols)


class ChannelDecoder(nn.Module):
    """Decode channel symbols back to semantic features."""

    def __init__(self, d_model: int = 128, hidden1: int = 256, hidden2: int = 128, symbol_dim: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.symbol_dim = symbol_dim
        self.fc1 = nn.Linear(symbol_dim, hidden2)
        self.nam1 = NoiseAdaptiveModulator(d_model=hidden2)
        self.fc2 = nn.Linear(hidden2, hidden1)
        self.nam2 = NoiseAdaptiveModulator(d_model=hidden1)
        self.out_proj = nn.Linear(hidden1, d_model)
        self.relu = nn.ReLU()

    def forward(self, received_symbols: Tensor, snr: Optional[Tensor] = None) -> Tensor:
        """Forward decode channel symbols [B,C] -> semantic code [B,128]."""
        if received_symbols.ndim != 2 or received_symbols.shape[-1] != self.symbol_dim:
            raise ValueError(
                f"Expected received_symbols shape [B,{self.symbol_dim}], got {tuple(received_symbols.shape)}"
            )
        batch_size = received_symbols.shape[0]
        if snr is None:
            snr = torch.zeros(batch_size, 1, device=received_symbols.device, dtype=received_symbols.dtype)
        else:
            snr = snr.to(device=received_symbols.device, dtype=received_symbols.dtype)

        x = self.fc1(received_symbols)
        x = self.nam1(snr, x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.nam2(snr, x)
        x = self.relu(x)

        return self.out_proj(x)


def power_normalize(symbols: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize average power on the last dimension."""
    power = torch.mean(symbols.pow(2), dim=-1, keepdim=True)
    return symbols / torch.sqrt(power + eps)
