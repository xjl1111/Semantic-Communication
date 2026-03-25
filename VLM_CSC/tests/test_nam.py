"""NAM shape tests template.

Expected:
- input r:[B,1], G:[B,T,128]
- output [B,T,128]
"""

import torch

from models.nam import NoiseAdaptiveModulator


def test_nam_shape_template() -> None:
    nam = NoiseAdaptiveModulator(d_model=128)
    r = torch.randn(2, 1)
    g = torch.randn(2, 16, 128)
    out = nam(r, g)
    assert out.shape == (2, 16, 128)

