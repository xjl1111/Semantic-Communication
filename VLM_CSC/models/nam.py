"""Noise-Adaptive Modulator (NAM).

Input:
- r: Tensor[B,1] snr descriptor
- G: Tensor[B,T,128] feature tensor

Output:
- A: Tensor[B,T,128]

Paper traceability:
- Core equations and 56/128/56/56 neuron counts are 论文明确写出.
- Dimensional reconciliation with d_model=128 is 为复现做的合理实现选择.
"""

import torch
from torch import Tensor, nn


class NoiseAdaptiveModulator(nn.Module):
    """SNR-conditioned feature gate module."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.snr_fc1 = nn.Linear(1, 56)
        self.snr_fc2 = nn.Linear(56, 128)
        self.snr_fc3 = nn.Linear(128, 56)

        self.feature_fc = nn.Linear(d_model, 56)
        self.out_fc = nn.Linear(56, d_model)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, r: Tensor, g: Tensor) -> Tensor:
        """Apply SNR-aware gating.

        Shapes:
            r: [B,1]
            g: [B,T,128]
            out: [B,T,128]
        """
        if r.ndim != 2 or r.shape[-1] != 1:
            raise ValueError(f"Expected r shape [B,1], got {tuple(r.shape)}")
        if g.ndim != 3 or g.shape[-1] != self.d_model:
            raise ValueError(f"Expected g shape [B,T,{self.d_model}], got {tuple(g.shape)}")

        v_prime = self.relu(self.snr_fc2(self.relu(self.snr_fc1(r))))
        v = self.sigmoid(self.snr_fc3(v_prime))

        e = self.feature_fc(g)
        k = self.sigmoid(e * v.unsqueeze(1))
        a_bottleneck = k * e
        out = self.out_fc(a_bottleneck)
        return out
