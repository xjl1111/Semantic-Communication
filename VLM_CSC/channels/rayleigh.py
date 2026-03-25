"""Rayleigh channel implementation scaffold.

Input/Output:
- input symbols: Tensor[B,T,C]
- output symbols: Tensor[B,T,C]

Paper traceability:
- Rayleigh channel for Fig.8 is 论文明确写出.
- Real-valued i.i.d coefficient sampling is 为复现做的合理实现选择.
"""

import torch
from torch import Tensor


def rayleigh_channel(symbols: Tensor, snr_db: float, training_mode: bool, seed: int = 42) -> Tensor:
    """Apply i.i.d. Rayleigh fading and additive noise."""
    if symbols.ndim != 3:
        raise ValueError(f"Expected symbols shape [B,T,C], got {tuple(symbols.shape)}")

    device = symbols.device
    dtype = symbols.dtype

    if training_mode:
        n1 = torch.randn_like(symbols)
        n2 = torch.randn_like(symbols)
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        n1 = torch.randn(symbols.shape, generator=generator, device=device, dtype=dtype)
        n2 = torch.randn(symbols.shape, generator=generator, device=device, dtype=dtype)

    fading = torch.sqrt((n1.pow(2) + n2.pow(2)) / 2.0)
    faded = fading * symbols

    snr_linear = 10.0 ** (snr_db / 10.0)
    signal_power = torch.mean(faded.pow(2), dim=(1, 2), keepdim=True)
    noise_power = signal_power / snr_linear
    noise_std = torch.sqrt(noise_power)

    if training_mode:
        noise = torch.randn_like(faded) * noise_std
    else:
        generator_noise = torch.Generator(device=device)
        generator_noise.manual_seed(seed + 1)
        noise = torch.randn(faded.shape, generator=generator_noise, device=device, dtype=dtype) * noise_std

    return faded + noise
