"""Semantic and Channel Decoder modules."""
from __future__ import annotations

import torch
import torch.nn as nn

from model.models.nam import NAM


class SemanticDecoder(nn.Module):
    """Transformer-based semantic decoder with optional NAM."""

    def __init__(
        self,
        feature_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        use_nam: bool = True,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.use_nam = bool(use_nam)
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=self.feature_dim,
                    nhead=int(num_heads),
                    dim_feedforward=int(ff_dim),
                    dropout=float(dropout),
                    batch_first=True,
                )
                for _ in range(int(num_layers))
            ]
        )
        if self.use_nam:
            self.nam_layers = nn.ModuleList([NAM(feature_dim=self.feature_dim) for _ in range(int(num_layers))])
        else:
            self.nam_layers = None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        snr: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tgt.dim() != 3 or memory.dim() != 3:
            raise ValueError("SemanticDecoder expects tgt and memory with shape (B, L, D).")
        if tgt.size(-1) != self.feature_dim or memory.size(-1) != self.feature_dim:
            raise ValueError(f"SemanticDecoder expects feature dim {self.feature_dim}.")

        out = tgt
        for i, transformer_layer in enumerate(self.transformer_layers):
            out = transformer_layer(
                tgt=out,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            if self.use_nam:
                out = self.nam_layers[i](out, snr=snr)
        return out


class ChannelDecoder(nn.Module):
    """FC-based channel decoder with optional NAM."""

    def __init__(self, input_dim: int = 128, output_dim: int = 128, use_nam: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.use_nam = bool(use_nam)

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc_out = nn.Linear(256, self.output_dim)
        self.relu = nn.ReLU()

        if self.use_nam:
            self.nam1 = NAM(feature_dim=128)
            self.nam2 = NAM(feature_dim=256)
        else:
            self.nam1 = None
            self.nam2 = None

    def forward(self, x: torch.Tensor, snr: float | torch.Tensor | None = None) -> torch.Tensor:
        if x.size(-1) != self.input_dim:
            raise ValueError(f"ChannelDecoder expects last dim {self.input_dim}, got {x.size(-1)}.")

        out = self.relu(self.fc1(x))
        if self.use_nam:
            out = self.nam1(out, snr=snr)
        out = self.relu(self.fc2(out))
        if self.use_nam:
            out = self.nam2(out, snr=snr)
        out = self.fc_out(out)
        return out
