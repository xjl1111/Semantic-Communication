import torch


def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean squared error over all elements."""
    return torch.mean((x - y) ** 2)


def avg_power(x: torch.Tensor) -> torch.Tensor:
    """Average power E[|x|^2] over all elements."""
    return torch.mean(x ** 2)
