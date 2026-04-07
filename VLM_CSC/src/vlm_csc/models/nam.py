"""Noise Attention Mechanism (NAM) module."""
from __future__ import annotations

import torch
import torch.nn as nn


class NAM(nn.Module):
    """Noise Attention Mechanism (NAM) - strict paper implementation.

    Paper specifies 4-layer FF with neuron counts: 56, 128, 56, 56.

    SNR projection path (layers 1-3):
        v' = ReLU(W_n2 · ReLU(W_n1 · r + b_n1) + b_n2)       [layers 1,2: 56, 128]
        v  = Sigmoid(W_n3 · v' + b_n3)                          [layer 3: 56]

    Feature transform (layer 4):
        e  = W_n4 · G + b_n4                                     [layer 4: 56]

    Gate & output:
        K  = Sigmoid(e · v)                                       [56-dim]
        gate = Linear(K, feature_dim)                             [back-project to feature_dim]
        A_i = Sigmoid(gate_i) · G_i
    """

    _PAPER_HIDDEN_DIMS = (56, 128, 56, 56)

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        d1, d2, d3, d4 = self._PAPER_HIDDEN_DIMS

        # SNR projection: 1 -> 56 (ReLU) -> 128 (ReLU) -> 56 (Sigmoid)
        self.snr_fc1 = nn.Linear(1, d1)
        self.snr_fc2 = nn.Linear(d1, d2)
        self.snr_fc3 = nn.Linear(d2, d3)

        # Feature transform: feature_dim -> 56
        self.feat_fc = nn.Linear(self.feature_dim, d4)

        # Back-projection: 56 -> feature_dim
        self.gate_proj = nn.Linear(d4, self.feature_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def _format_snr(self, x: torch.Tensor, snr: float | torch.Tensor | None) -> torch.Tensor:
        batch_size = x.size(0)
        if snr is None:
            snr_tensor = torch.zeros((batch_size, 1), device=x.device, dtype=x.dtype)
        elif isinstance(snr, torch.Tensor):
            snr_tensor = snr.to(device=x.device, dtype=x.dtype)
            if snr_tensor.dim() == 0:
                snr_tensor = snr_tensor.view(1, 1).expand(batch_size, 1)
            elif snr_tensor.dim() == 1:
                snr_tensor = snr_tensor.view(-1, 1)
            if snr_tensor.size(0) == 1 and batch_size > 1:
                snr_tensor = snr_tensor.expand(batch_size, 1)
        else:
            snr_tensor = torch.full((batch_size, 1), float(snr), device=x.device, dtype=x.dtype)
        return snr_tensor

    def forward(self, x: torch.Tensor, snr: float | torch.Tensor | None = None) -> torch.Tensor:
        snr_tensor = self._format_snr(x, snr)
        v_prime = self.relu(self.snr_fc1(snr_tensor))
        v_prime = self.relu(self.snr_fc2(v_prime))
        v = self.sigmoid(self.snr_fc3(v_prime))
        if x.dim() == 3:
            v = v.unsqueeze(1)

        e = self.feat_fc(x)
        K = self.sigmoid(e * v)
        gate = self.sigmoid(self.gate_proj(K))
        return x * gate
