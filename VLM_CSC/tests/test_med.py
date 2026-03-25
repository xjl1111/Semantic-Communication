"""MED behavior tests template.

Expected:
- RBF similarity in [0,1]
- transfer triggered when STM capacity is reached
"""

import torch

from models.med import MemoryEnhancedDistillation, MemoryItem


def test_rbf_range_template() -> None:
    med = MemoryEnhancedDistillation(stm_max_size=3, tau=10.0, threshold=0.05)
    a = torch.randn(128)
    b = torch.randn(128)
    sim = med.rbf_similarity(a, b)
    assert 0.0 <= float(sim.item()) <= 1.0


def test_transfer_template() -> None:
    med = MemoryEnhancedDistillation(stm_max_size=2, tau=10.0, threshold=0.05, transfer_if="greater")
    item1 = MemoryItem("text1", torch.zeros(128), "task1", "id1", 1)
    item2 = MemoryItem("text2", torch.zeros(128), "task1", "id2", 2)
    med.add_to_stm(item1)
    med.add_to_stm(item2)
    assert len(med.ltm) >= 1

