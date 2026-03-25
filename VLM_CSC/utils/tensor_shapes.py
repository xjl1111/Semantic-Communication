"""Tensor shape assertion helpers for fast debugging."""

from torch import Tensor


def assert_shape(tensor: Tensor, expected_rank: int, name: str) -> None:
    """Assert tensor rank and raise readable error message."""
    if tensor.ndim != expected_rank:
        raise ValueError(f"{name} rank mismatch: got {tensor.ndim}, expected {expected_rank}")
