"""Shape tests template for end-to-end module contracts."""

import torch

from utils.tensor_shapes import assert_shape


def test_semantic_shape_template() -> None:
    """Template: token ids [B,T] -> semantic [B,T,128]."""
    fake_semantic = torch.randn(2, 8, 128)
    assert_shape(fake_semantic, expected_rank=3, name="semantic_features")
    assert fake_semantic.shape[-1] == 128


def test_channel_shape_template() -> None:
    """Template: [B,T,128] -> channel -> [B,T,128]."""
    channel_io = torch.randn(2, 8, 128)
    assert_shape(channel_io, expected_rank=3, name="channel_features")
    assert channel_io.shape == (2, 8, 128)
