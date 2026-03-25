"""AWGN channel implementation scaffold.

Input/Output:
- input symbols: Tensor[B,T,C]
- output symbols: Tensor[B,T,C]

Paper traceability:
- AWGN channel is 论文明确写出.
- Real-valued implementation details are 为复现做的合理实现选择.
"""

import torch
from torch import Tensor


def awgn_channel(symbols: Tensor, snr_db: float, training_mode: bool, seed: int = 42) -> Tensor:
    """Apply AWGN noise under target SNR."""
    if symbols.ndim not in (2, 3):
        raise ValueError(f"Expected symbols shape [B,C] or [B,T,C], got {tuple(symbols.shape)}")

    device = symbols.device
    dtype = symbols.dtype
    snr_linear = 10.0 ** (snr_db / 10.0)

    reduce_dims = tuple(range(1, symbols.ndim))
    signal_power = torch.mean(symbols.pow(2), dim=reduce_dims, keepdim=True)
    noise_power = signal_power / snr_linear
    noise_std = torch.sqrt(noise_power)

    if training_mode:
        noise = torch.randn_like(symbols) * noise_std
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        noise = torch.randn(symbols.shape, generator=generator, device=device, dtype=dtype) * noise_std

    return symbols + noise
