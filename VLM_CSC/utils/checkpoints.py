"""Checkpoint save/load utilities."""

from pathlib import Path

import torch


def save_checkpoint(path: Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path) -> dict:
    path = Path(path)
    return torch.load(path, map_location="cpu")
