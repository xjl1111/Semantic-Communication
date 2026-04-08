"""Base class for receiver CKB modules."""
from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class ReceiverCKB(ABC):
    """Abstract base class for receiver CKB components."""

    @abstractmethod
    def forward(self, text: str, **kwargs) -> Image.Image:
        """Generate image from text description."""
        ...
