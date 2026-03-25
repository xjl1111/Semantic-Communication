"""Minimal tests template for AWGN/Rayleigh channel modules."""

import torch

from channels.awgn import awgn_channel
from channels.rayleigh import rayleigh_channel


def test_awgn_template() -> None:
    x = torch.randn(2, 8, 128)
    y = awgn_channel(x, snr_db=4.0, training_mode=False, seed=7)
    assert y.shape == x.shape


def test_rayleigh_template() -> None:
    x = torch.randn(2, 8, 128)
    y = rayleigh_channel(x, snr_db=4.0, training_mode=False, seed=7)
    assert y.shape == x.shape

