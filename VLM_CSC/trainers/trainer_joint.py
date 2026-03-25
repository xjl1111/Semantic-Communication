"""Stage C crossover-based iterative trainer.

Paper traceability:
- Alternating Stage A/B iterative optimization is 论文明确写出.
- Early-stopping rule is 为复现做的合理实现选择.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class JointTrainConfig:
    max_rounds: int = 10
    patience: int = 3


class JointTrainer:
    """Alternating trainer for stage A and B."""

    def __init__(self, channel_trainer: object, semantic_trainer: object, config: JointTrainConfig | None = None) -> None:
        self.channel_trainer = channel_trainer
        self.semantic_trainer = semantic_trainer
        self.config = config or JointTrainConfig()

    def fit(self) -> dict:
        history: List[Dict[str, float]] = []
        best_score = float("inf")
        best_round = -1
        stale_rounds = 0

        for round_idx in range(self.config.max_rounds):
            channel_train = self.channel_trainer.train_one_epoch()
            channel_val = self.channel_trainer.validate()

            semantic_train = self.semantic_trainer.train_one_epoch()
            semantic_val = self.semantic_trainer.validate()

            combined_val = float(channel_val["loss"]) + float(semantic_val["loss"])
            record = {
                "round": float(round_idx),
                "channel_train_loss": float(channel_train["loss"]),
                "channel_val_loss": float(channel_val["loss"]),
                "semantic_train_loss": float(semantic_train["loss"]),
                "semantic_val_loss": float(semantic_val["loss"]),
                "combined_val_loss": combined_val,
            }
            history.append(record)

            if combined_val < best_score:
                best_score = combined_val
                best_round = round_idx
                stale_rounds = 0
            else:
                stale_rounds += 1

            if stale_rounds >= self.config.patience:
                break

        return {
            "best_round": best_round,
            "best_combined_val_loss": best_score,
            "history": history,
        }
