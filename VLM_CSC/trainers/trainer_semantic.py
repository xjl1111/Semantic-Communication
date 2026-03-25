"""Stage B trainer: semantic encoder/decoder (+NAM) with channel in loop.

Paper traceability:
- Stage-wise training and CE objective are 论文明确写出.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable

import torch
from torch import nn

from losses.text_ce import text_cross_entropy


class SemanticTrainer:
    """Trainer scaffold for semantic coding stage."""

    def __init__(
        self,
        semantic_encoder: nn.Module,
        semantic_decoder: nn.Module,
        channel_encoder: nn.Module,
        channel_decoder: nn.Module,
        dataloader: Iterable[Dict[str, torch.Tensor]],
        channel_fn: Callable[[torch.Tensor, float, bool, int], torch.Tensor],
        device: str = "cpu",
        lr: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self.semantic_encoder = semantic_encoder.to(device)
        self.semantic_decoder = semantic_decoder.to(device)
        self.channel_encoder = channel_encoder.to(device)
        self.channel_decoder = channel_decoder.to(device)
        self.dataloader = dataloader
        self.channel_fn = channel_fn
        self.device = device
        self.seed = seed

        for module in (self.channel_encoder, self.channel_decoder):
            for p in module.parameters():
                p.requires_grad = False
            module.eval()

        self.semantic_encoder.train()
        self.semantic_decoder.train()
        params = list(self.semantic_encoder.parameters()) + list(self.semantic_decoder.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr)

    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            out[key] = value.to(self.device) if isinstance(value, torch.Tensor) else value
        return out

    @staticmethod
    def _shift_for_decoder(token_ids: torch.Tensor, attention_mask: torch.Tensor, bos_id: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build decoder input/target with standard autoregressive shift.

        decoder_input_ids: [BOS, w1, ..., w_{L-1}]
        targets:           [w1,  ..., w_L]
        """
        if token_ids.shape[1] < 2:
            raise ValueError("token sequence length must be >= 2 for shifted decoder training")

        decoder_input = token_ids[:, :-1].clone()
        decoder_input[:, 0] = bos_id
        targets = token_ids[:, 1:].contiguous()
        target_mask = attention_mask[:, 1:].contiguous()
        return decoder_input, targets, target_mask

    def train_one_epoch(self) -> dict:
        self.semantic_encoder.train()
        self.semantic_decoder.train()
        for module in (self.semantic_encoder, self.semantic_decoder):
            for p in module.parameters():
                p.requires_grad = True
        for module in (self.channel_encoder, self.channel_decoder):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
        total_loss = 0.0
        steps = 0

        for raw_batch in self.dataloader:
            batch = self._batch_to_device(raw_batch)
            token_ids = batch["token_ids"].long()
            attn_mask = batch.get("attention_mask", torch.ones_like(token_ids))
            snr = batch.get("snr", torch.zeros(token_ids.shape[0], 1, device=self.device))
            snr_db = float(batch.get("snr_db", 4.0))

            decoder_input_ids, targets, target_mask = self._shift_for_decoder(token_ids, attn_mask)

            semantic_features = self.semantic_encoder(token_ids=token_ids, snr=snr)
            symbols = self.channel_encoder(semantic_features=semantic_features, snr=snr)
            received = self.channel_fn(symbols, snr_db=snr_db, training_mode=True, seed=self.seed)
            decoded_features = self.channel_decoder(received_symbols=received, snr=snr)
            logits = self.semantic_decoder(
                channel_features=decoded_features,
                target_ids=decoder_input_ids,
                snr=snr,
            )
            loss = text_cross_entropy(logits=logits, targets=targets, attention_mask=target_mask)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        mean_loss = total_loss / max(steps, 1)
        return {"loss": mean_loss, "steps": steps}

    def validate(self) -> dict:
        self.semantic_encoder.eval()
        self.semantic_decoder.eval()
        total_loss = 0.0
        steps = 0

        with torch.no_grad():
            for raw_batch in self.dataloader:
                batch = self._batch_to_device(raw_batch)
                token_ids = batch["token_ids"].long()
                attn_mask = batch.get("attention_mask", torch.ones_like(token_ids))
                snr = batch.get("snr", torch.zeros(token_ids.shape[0], 1, device=self.device))
                snr_db = float(batch.get("snr_db", 4.0))

                decoder_input_ids, targets, target_mask = self._shift_for_decoder(token_ids, attn_mask)

                semantic_features = self.semantic_encoder(token_ids=token_ids, snr=snr)
                symbols = self.channel_encoder(semantic_features=semantic_features, snr=snr)
                received = self.channel_fn(symbols, snr_db=snr_db, training_mode=False, seed=self.seed)
                decoded_features = self.channel_decoder(received_symbols=received, snr=snr)
                logits = self.semantic_decoder(
                    channel_features=decoded_features,
                    target_ids=decoder_input_ids,
                    snr=snr,
                )
                loss = text_cross_entropy(logits=logits, targets=targets, attention_mask=target_mask)

                total_loss += float(loss.item())
                steps += 1

        mean_loss = total_loss / max(steps, 1)
        return {"loss": mean_loss, "steps": steps}
