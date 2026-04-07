"""Base class for sender CKB modules."""
from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image
import torch


class SenderCKB(ABC):
    """Abstract base class for sender CKB (Caption Knowledge Base) components."""

    @abstractmethod
    def forward(self, image: Image.Image | torch.Tensor) -> str:
        """Generate caption/tags from image."""
        ...
