"""Memory-Enhanced Distillation (MED) storage and sampling.

Paper traceability:
- STM/LTM, tau=10, STM size=500, threshold=0.05 are 论文明确写出.
- Sampling ratio and transfer conflict switch are 为复现做的合理实现选择.
"""

from dataclasses import dataclass
import random
from typing import List, Literal

import torch
from torch import Tensor


@dataclass
class MemoryItem:
    semantic_text: str
    semantic_feature: Tensor
    task_id: str
    sample_id: str
    timestamp: int


class MemoryEnhancedDistillation:
    """STM/LTM manager with RBF-based transfer rule."""

    def __init__(
        self,
        stm_max_size: int = 500,
        tau: float = 10.0,
        threshold: float = 0.05,
        transfer_if: Literal["greater", "smaller"] = "greater",
    ) -> None:
        self.stm_max_size = stm_max_size
        self.tau = tau
        self.threshold = threshold
        self.transfer_if = transfer_if
        self.stm: List[MemoryItem] = []
        self.ltm: List[MemoryItem] = []

    def add_to_stm(self, item: MemoryItem) -> None:
        self.stm.append(item)
        if len(self.stm) >= self.stm_max_size:
            self.transfer_from_stm_to_ltm()

    def rbf_similarity(self, feature_stm: Tensor, feature_ltm: Tensor) -> Tensor:
        a = feature_stm.float().reshape(-1)
        b = feature_ltm.float().reshape(-1)
        if a.shape != b.shape:
            min_dim = min(a.shape[0], b.shape[0])
            a = a[:min_dim]
            b = b[:min_dim]
        dist2 = torch.sum((a - b).pow(2))
        sim = torch.exp(-dist2 / (2.0 * (self.tau ** 2)))
        return sim.clamp(0.0, 1.0)

    def average_similarity(self, feature_stm: Tensor) -> Tensor:
        if len(self.ltm) == 0:
            return torch.tensor(0.0)
        sims = [self.rbf_similarity(feature_stm, item.semantic_feature) for item in self.ltm]
        return torch.stack(sims).mean()

    def transfer_from_stm_to_ltm(self) -> List[MemoryItem]:
        transferred: List[MemoryItem] = []
        if len(self.stm) == 0:
            return transferred

        remaining_stm: List[MemoryItem] = []
        for item in self.stm:
            avg_sim = self.average_similarity(item.semantic_feature)
            if self.transfer_if == "greater":
                should_transfer = bool(avg_sim > self.threshold)
            else:
                should_transfer = bool(avg_sim < self.threshold)

            if len(self.ltm) == 0:
                should_transfer = True

            if should_transfer:
                self.ltm.append(item)
                transferred.append(item)
            else:
                remaining_stm.append(item)

        self.stm = remaining_stm
        return transferred

    def sample_mixed_batch(self, batch_size: int, stm_ratio: float = 0.5) -> List[MemoryItem]:
        if batch_size <= 0:
            return []

        n_stm = int(round(batch_size * stm_ratio))
        n_ltm = batch_size - n_stm

        stm_pool = self.stm[:] if len(self.stm) > 0 else []
        ltm_pool = self.ltm[:] if len(self.ltm) > 0 else []

        sampled: List[MemoryItem] = []
        if len(stm_pool) > 0:
            sampled.extend(random.choices(stm_pool, k=n_stm))
        if len(ltm_pool) > 0:
            sampled.extend(random.choices(ltm_pool, k=n_ltm))

        while len(sampled) < batch_size:
            if len(stm_pool) > 0:
                sampled.append(random.choice(stm_pool))
            elif len(ltm_pool) > 0:
                sampled.append(random.choice(ltm_pool))
            else:
                break

        random.shuffle(sampled)
        return sampled[:batch_size]
