"""Stage A trainer: channel encoder/decoder (+NAM) with proxy loss.

Paper traceability:
- Stage-wise channel optimization intent is 论文明确写出.
- Proxy loss details are 为复现做的合理实现选择.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import torch
from torch import nn

from losses.mutual_info_proxy import channel_proxy_loss


class ChannelTrainer:
    """Trainer scaffold for channel coding stage."""

    def __init__(
        self,
        semantic_encoder: nn.Module,
        channel_encoder: nn.Module,
        channel_decoder: nn.Module,
        dataloader: Iterable[Dict[str, torch.Tensor]],
        channel_fn: Callable[[torch.Tensor, float, bool, int], torch.Tensor],
        device: str = "cpu",
        lr: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self.semantic_encoder = semantic_encoder.to(device)
        self.channel_encoder = channel_encoder.to(device)
        self.channel_decoder = channel_decoder.to(device)
        self.dataloader = dataloader
        self.channel_fn = channel_fn
        self.device = device
        self.seed = seed

        for p in self.semantic_encoder.parameters():
            p.requires_grad = False
        self.semantic_encoder.eval()

        self.channel_encoder.train()
        self.channel_decoder.train()
        params = list(self.channel_encoder.parameters()) + list(self.channel_decoder.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr)

    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            out[key] = value.to(self.device) if isinstance(value, torch.Tensor) else value
        return out

    def train_one_epoch(self) -> dict:
        self.channel_encoder.train()
        self.channel_decoder.train()
        for module in (self.channel_encoder, self.channel_decoder):
            for p in module.parameters():
                p.requires_grad = True
        total_loss = 0.0
        steps = 0

        for raw_batch in self.dataloader:
            batch = self._batch_to_device(raw_batch)
            token_ids = batch["token_ids"].long()
            snr = batch.get("snr", torch.zeros(token_ids.shape[0], 1, device=self.device))
            snr_db = float(batch.get("snr_db", 4.0))

            with torch.no_grad():
                semantic_features = self.semantic_encoder.encode_sentence(token_ids=token_ids, snr=snr)

            encoded_symbols = self.channel_encoder(semantic_features=semantic_features, snr=snr)
            received_symbols = self.channel_fn(encoded_symbols, snr_db=snr_db, training_mode=True, seed=self.seed)
            decoded_features = self.channel_decoder(received_symbols=received_symbols, snr=snr)

            loss = channel_proxy_loss(
                decoded_feat=decoded_features,
                original_feat=semantic_features,
                encoded_symbols=encoded_symbols,
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        mean_loss = total_loss / max(steps, 1)
        return {"loss": mean_loss, "steps": steps}

    def validate(self) -> dict:
        self.channel_encoder.eval()
        self.channel_decoder.eval()
        total_loss = 0.0
        steps = 0

        with torch.no_grad():
            for raw_batch in self.dataloader:
                batch = self._batch_to_device(raw_batch)
                token_ids = batch["token_ids"].long()
                snr = batch.get("snr", torch.zeros(token_ids.shape[0], 1, device=self.device))
                snr_db = float(batch.get("snr_db", 4.0))

                semantic_features = self.semantic_encoder.encode_sentence(token_ids=token_ids, snr=snr)
                encoded_symbols = self.channel_encoder(semantic_features=semantic_features, snr=snr)
                received_symbols = self.channel_fn(encoded_symbols, snr_db=snr_db, training_mode=False, seed=self.seed)
                decoded_features = self.channel_decoder(received_symbols=received_symbols, snr=snr)

                loss = channel_proxy_loss(
                    decoded_feat=decoded_features,
                    original_feat=semantic_features,
                    encoded_symbols=encoded_symbols,
                )
                total_loss += float(loss.item())
                steps += 1

        mean_loss = total_loss / max(steps, 1)
        return {"loss": mean_loss, "steps": steps}
