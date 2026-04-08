"""Memory Evolution and Distillation (MED) module."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class MemorySample:
    """A single memory sample stored in MED."""
    image_id: str
    caption_text: str
    token_ids: torch.Tensor
    semantic_feature: torch.Tensor
    timestamp: int
    dataset_id: str


class MED:
    """Memory Evolution and Distillation for continual learning.

    Manages short-term memory (STM) and long-term memory (LTM)
    with RBF similarity-based transfer mechanism.
    """

    def __init__(
        self,
        stm_max_size: int = 500,
        tau: float = 10.0,
        threshold: float = 0.05,
        transfer_if: str = "greater",
        ltm_max_size: Optional[int] = None,
        strict_paper_repro: bool = True,
    ):
        self.strict_paper_repro = bool(strict_paper_repro)

        if self.strict_paper_repro:
            if int(stm_max_size) != 500:
                raise RuntimeError(f"MED strict violation: stm_max_size must be 500, got {stm_max_size}")
            if abs(float(threshold) - 0.05) > 1e-12:
                raise RuntimeError(f"MED strict violation: threshold must be 0.05, got {threshold}")
            if abs(float(tau) - 10.0) > 1e-12:
                raise RuntimeError(f"MED strict violation: tau must be 10.0, got {tau}")

        self.stm_max_size = int(stm_max_size)
        self.tau = float(tau)
        self.threshold = float(threshold)
        self.transfer_if = str(transfer_if).lower()
        self.ltm_max_size = ltm_max_size if ltm_max_size is None else int(ltm_max_size)

        if self.transfer_if not in ("greater", "smaller"):
            raise ValueError("transfer_if must be 'greater' or 'smaller'.")

        self.stm: List[MemorySample] = []
        self.ltm: List[MemorySample] = []
        self._timestamp_counter = 0
        self._seen_sample_keys: set[tuple[str, str]] = set()

    @property
    def stm_samples(self) -> List[MemorySample]:
        return self.stm

    @property
    def ltm_samples(self) -> List[MemorySample]:
        return self.ltm

    def append_to_stm(self, sample: Dict) -> None:
        required_keys = {"image_id", "caption_text", "token_ids", "semantic_feature", "dataset_id"}
        if not required_keys.issubset(set(sample.keys())):
            missing = required_keys - set(sample.keys())
            raise ValueError(f"Missing sample fields for MED: {missing}")

        token_ids = sample["token_ids"]
        feature = sample["semantic_feature"]
        if not isinstance(token_ids, torch.Tensor) or not isinstance(feature, torch.Tensor):
            raise ValueError("token_ids and semantic_feature must be torch.Tensor.")

        image_id = str(sample["image_id"])
        dataset_id = str(sample["dataset_id"])
        key = (dataset_id, image_id)
        if key in self._seen_sample_keys:
            raise RuntimeError(
                f"MED duplicate sample rejected for key=(dataset_id={dataset_id}, image_id={image_id})"
            )

        item = MemorySample(
            image_id=image_id,
            caption_text=str(sample["caption_text"]),
            token_ids=token_ids.detach().cpu(),
            semantic_feature=feature.detach().cpu().float(),
            timestamp=self._timestamp_counter,
            dataset_id=dataset_id,
        )
        self._timestamp_counter += 1
        self.stm.append(item)
        self._seen_sample_keys.add(key)

    def add_to_stm(self, sample: Dict) -> None:
        self.append_to_stm(sample)

    def is_stm_full(self) -> bool:
        return len(self.stm) >= self.stm_max_size

    def compute_rbf(self, stm_feat: torch.Tensor, ltm_feat: torch.Tensor) -> torch.Tensor:
        diff2 = (stm_feat - ltm_feat).pow(2).sum()
        return torch.exp(-diff2 / (2 * (self.tau ** 2)))

    def compute_avg_similarity(self, stm_feat: torch.Tensor) -> float:
        if len(self.ltm) == 0:
            return 0.0

        scores = []
        for item in self.ltm:
            score = self.compute_rbf(stm_feat, item.semantic_feature)
            scores.append(score)
        stacked = torch.stack(scores)
        return float(stacked.mean().item())

    def select_representatives(self) -> List[MemorySample]:
        """Batch matrix computation for STM->LTM transfer selection."""
        if len(self.stm) == 0:
            return []

        if len(self.ltm) == 0:
            return list(self.stm)

        stm_feats = torch.stack([item.semantic_feature for item in self.stm])
        ltm_feats = torch.stack([item.semantic_feature for item in self.ltm])

        diff2 = torch.cdist(stm_feats, ltm_feats, p=2).pow(2)
        rbf = torch.exp(-diff2 / (2.0 * (self.tau ** 2)))
        avg_sim = rbf.mean(dim=1).tolist()

        selected: List[MemorySample] = []
        for item, sim in zip(self.stm, avg_sim):
            if self.transfer_if == "greater":
                cond = sim > self.threshold
            else:
                cond = sim < self.threshold
            if cond:
                selected.append(item)

        import numpy as _np
        _sims = _np.array(avg_sim)
        print(
            f"[MED] select: {len(selected)}/{len(self.stm)} passed "
            f"(threshold={self.threshold}, transfer_if={self.transfer_if})  "
            f"sim: min={_sims.min():.4f} mean={_sims.mean():.4f} max={_sims.max():.4f}  "
            f"LTM_before={len(self.ltm)}"
        )
        return selected

    def maybe_transfer_to_ltm(self) -> Dict[str, int]:
        if not self.is_stm_full():
            return {"triggered": 0, "moved": 0, "stm_size": len(self.stm), "ltm_size": len(self.ltm)}

        selected = self.select_representatives()
        if len(selected) == 0:
            self.clear_stm()
            return {"triggered": 1, "moved": 0, "stm_size": len(self.stm), "ltm_size": len(self.ltm)}

        self.ltm.extend(selected)
        if self.ltm_max_size is not None and len(self.ltm) > self.ltm_max_size:
            self.ltm = self.ltm[-self.ltm_max_size :]

        self.clear_stm()
        return {"triggered": 1, "moved": len(selected), "stm_size": len(self.stm), "ltm_size": len(self.ltm)}

    def clear_stm(self) -> None:
        self.stm.clear()

    def sample_train_batch(self, batch_size: int, stm_ratio: float = 0.5) -> List[MemorySample]:
        batch_size = int(batch_size)
        if batch_size <= 0:
            return []

        stm_count = int(round(batch_size * float(stm_ratio)))
        stm_count = max(0, min(stm_count, batch_size))
        ltm_count = batch_size - stm_count

        stm_samples = random.sample(self.stm, min(stm_count, len(self.stm))) if len(self.stm) > 0 else []
        ltm_samples = random.sample(self.ltm, min(ltm_count, len(self.ltm))) if len(self.ltm) > 0 else []

        merged = stm_samples + ltm_samples
        if len(merged) < batch_size:
            pool = self.stm + self.ltm
            remain = batch_size - len(merged)
            if len(pool) > 0:
                merged.extend(random.sample(pool, min(remain, len(pool))))
        return merged

    def maybe_update(self) -> Dict[str, int]:
        return self.maybe_transfer_to_ltm()
