"""Sender CKB modules."""
from vlm_csc.senders.base import SenderCKB
from vlm_csc.senders.blip import SenderCKB_BLIP
from vlm_csc.senders.ram import SenderCKB_RAM

__all__ = ["SenderCKB", "SenderCKB_BLIP", "SenderCKB_RAM"]
