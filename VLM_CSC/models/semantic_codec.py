"""Semantic encoder/decoder with NAM interleaving.

Input/Output shapes:
- semantic encoder input embedding: Tensor[B, T, 128]
- semantic encoder output: Tensor[B, T, 128]
- semantic decoder input: Tensor[B, T, 128]
- semantic decoder output logits: Tensor[B, T, vocab_size]

Paper traceability:
- 3 layers, 8 heads, d_model=128 and NAM interleaving are 论文明确写出.
- Positional embedding/dropout/FFN ratio are 为复现做的合理实现选择.
"""

from typing import Optional

import torch
from torch import Tensor, nn

from .nam import NoiseAdaptiveModulator


class SemanticEncoder(nn.Module):
    """Transformer encoder stack for semantic coding."""

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 8, num_layers: int = 3) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.max_len = 512

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(self.max_len, d_model)
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=4 * d_model,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.nams = nn.ModuleList([NoiseAdaptiveModulator(d_model=d_model) for _ in range(num_layers)])

    def forward(self, token_ids: Tensor, snr: Optional[Tensor] = None) -> Tensor:
        """Encode token ids [B,T] to semantic features [B,T,128]."""
        if token_ids.ndim != 2:
            raise ValueError(f"Expected token_ids shape [B,T], got {tuple(token_ids.shape)}")
        batch_size, seq_len = token_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}")

        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        if snr is None:
            snr = torch.zeros(batch_size, 1, device=token_ids.device, dtype=x.dtype)
        else:
            snr = snr.to(device=token_ids.device, dtype=x.dtype)

        for layer, nam in zip(self.layers, self.nams):
            x = layer(x)
            x = nam(snr, x)
        return x


class SemanticDecoder(nn.Module):
    """Transformer decoder stack for semantic decoding."""

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 8, num_layers: int = 3) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.max_len = 512

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(self.max_len, d_model)
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=4 * d_model,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.nams = nn.ModuleList([NoiseAdaptiveModulator(d_model=d_model) for _ in range(num_layers)])
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, channel_features: Tensor, target_ids: Optional[Tensor] = None, snr: Optional[Tensor] = None) -> Tensor:
        """Decode [B,T,128] to logits [B,T,vocab_size]."""
        if channel_features.ndim != 3 or channel_features.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected channel_features shape [B,T,{self.d_model}], got {tuple(channel_features.shape)}"
            )
        batch_size, seq_len, _ = channel_features.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}")

        if target_ids is None:
            tgt_ids = torch.zeros(batch_size, seq_len, device=channel_features.device, dtype=torch.long)
        else:
            if target_ids.ndim != 2:
                raise ValueError(f"Expected target_ids shape [B,T], got {tuple(target_ids.shape)}")
            tgt_ids = target_ids
            seq_len = tgt_ids.shape[1]

        positions = torch.arange(seq_len, device=channel_features.device).unsqueeze(0).expand(batch_size, seq_len)
        tgt = self.token_embedding(tgt_ids) + self.position_embedding(positions)
        tgt = self.dropout(tgt)

        if snr is None:
            snr = torch.zeros(batch_size, 1, device=channel_features.device, dtype=channel_features.dtype)
        else:
            snr = snr.to(device=channel_features.device, dtype=channel_features.dtype)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=channel_features.device, dtype=torch.bool),
            diagonal=1,
        )

        x = tgt
        for layer, nam in zip(self.layers, self.nams):
            x = layer(tgt=x, memory=channel_features, tgt_mask=causal_mask)
            x = nam(snr, x)

        logits = self.output_proj(x)
        return logits

    @torch.no_grad()
    def greedy_decode(self, channel_features: Tensor, bos_id: int, eos_id: int, max_len: int) -> Tensor:
        """Greedy decoding output token ids [B,T]."""
        if channel_features.ndim != 3:
            raise ValueError(f"Expected channel_features shape [B,T,D], got {tuple(channel_features.shape)}")

        batch_size = channel_features.shape[0]
        decoded = torch.full(
            (batch_size, 1),
            fill_value=bos_id,
            device=channel_features.device,
            dtype=torch.long,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=channel_features.device)

        for _ in range(max_len - 1):
            logits = self.forward(channel_features=channel_features, target_ids=decoded)
            next_token = logits[:, -1, :].argmax(dim=-1)
            decoded = torch.cat([decoded, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_id)
            if torch.all(finished):
                break
        return decoded
