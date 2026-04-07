"""Physical channel model (AWGN, Rayleigh)."""
from __future__ import annotations

import torch
import torch.nn as nn


class PhysicalChannel(nn.Module):
    """Physical channel model.

    Supports AWGN and Rayleigh fading channels with power normalization.
    """

    def __init__(self, channel_type: str = "awgn", rayleigh_mode: str = "fast"):
        super().__init__()
        self.channel_type = str(channel_type).lower()
        self.rayleigh_mode = str(rayleigh_mode).lower()

    @staticmethod
    def power_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        power = x.pow(2).mean(dim=-1, keepdim=True)
        return x / torch.sqrt(power + eps)

    @staticmethod
    def _snr_to_std(snr_db: float) -> float:
        snr_linear = 10.0 ** (float(snr_db) / 10.0)
        sigma2 = 1.0 / snr_linear
        return float(sigma2 ** 0.5)

    def _awgn(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        std = self._snr_to_std(snr_db)
        noise = torch.randn_like(x) * std
        return x + noise

    def _rayleigh(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        std = self._snr_to_std(snr_db)
        if self.rayleigh_mode not in ("fast", "block"):
            raise ValueError("rayleigh_mode must be 'fast' or 'block'.")

        if self.rayleigh_mode == "fast":
            h_real = torch.randn_like(x)
            h_imag = torch.randn_like(x)
            h = torch.sqrt(h_real.pow(2) + h_imag.pow(2)) / (2.0 ** 0.5)
        else:
            shape = [x.size(0)] + [1] * (x.dim() - 1)
            h_real = torch.randn(shape, device=x.device, dtype=x.dtype)
            h_imag = torch.randn(shape, device=x.device, dtype=x.dtype)
            h = (torch.sqrt(h_real.pow(2) + h_imag.pow(2)) / (2.0 ** 0.5)).expand_as(x)

        noise = torch.randn_like(x) * std
        return h * x + noise

    def forward(self, x: torch.Tensor, snr_db: float, normalize_power: bool = True) -> torch.Tensor:
        if self.channel_type == "rayleigh" and not normalize_power:
            raise RuntimeError("Rayleigh channel requires normalize_power=True under strict protocol.")

        if normalize_power:
            x = self.power_normalize(x)

        if self.channel_type == "awgn":
            return self._awgn(x, snr_db)
        if self.channel_type == "rayleigh":
            return self._rayleigh(x, snr_db)

        raise ValueError("channel_type must be 'awgn' or 'rayleigh'.")
