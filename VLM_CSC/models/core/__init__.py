"""Core VLM-CSC modules (non-baseline).

This namespace groups primary modules used by the paper's main pipeline:
- BLIP/SD knowledge bases
- semantic/channel codecs
- NAM and MED
"""

from models.kb_blip import BlipKnowledgeBase
from models.kb_sd import StableDiffusionKnowledgeBase
from models.semantic_codec import SemanticEncoder, SemanticDecoder
from models.channel_codec import ChannelEncoder, ChannelDecoder
from models.nam import NoiseAdaptiveModulator
from models.med import MemoryEnhancedDistillation, MemoryItem

__all__ = [
    "BlipKnowledgeBase",
    "StableDiffusionKnowledgeBase",
    "SemanticEncoder",
    "SemanticDecoder",
    "ChannelEncoder",
    "ChannelDecoder",
    "NoiseAdaptiveModulator",
    "MemoryEnhancedDistillation",
    "MemoryItem",
]
