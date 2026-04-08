"""VLMCscSystem - Main VLM-CSC system model."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from model.models.encoders import SemanticEncoder, ChannelEncoder
from model.models.decoders import SemanticDecoder, ChannelDecoder
from model.models.channel import PhysicalChannel
from model.models.memory import MED
from model.senders import SenderCKB_BLIP, SenderCKB_RAM
from model.receivers import ReceiverCKB_SD
from model.tokenization import SimpleTextTokenizer


class VLMCscSystem(nn.Module):
    """VLM-CSC system model.

    Communication path follows sequence-level transport (seq -> seq):
    semantic token features are transmitted through the channel without sentence-level pooling.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        max_text_len: int = 32,
        vocab_size: int = 30522,
        channel_type: str = "awgn",
        sender_type: str = "blip",
        use_real_ckb: bool = False,
        use_real_receiver_ckb: Optional[bool] = None,
        enable_med: bool = False,
        med_kwargs: Optional[Dict] = None,
        blip_dir: str | Path = "./data/assets/downloaded_models/blip",
        ram_ckpt: str | Path = "./data/assets/downloaded_models/ram_swin_large_14m.pth",
        sd_dir: str | Path = "./data/assets/downloaded_models/sd15",
        device: Optional[str] = None,
        use_nam: bool = True,
        channel_dim: int | None = None,
        caption_mode: str = "baseline",
        caption_prompt: str | None = None,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.channel_dim = int(channel_dim) if channel_dim is not None else self.feature_dim
        self.max_text_len = int(max_text_len)
        self.vocab_size = int(vocab_size)
        self.use_nam = bool(use_nam)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.runtime_device = device
        self.sender_type = str(sender_type).lower()

        if self.sender_type == "blip":
            self.sender_ckb = SenderCKB_BLIP(
                blip_dir=blip_dir,
                use_real_ckb=use_real_ckb,
                device=self.runtime_device,
                caption_mode=caption_mode,
                caption_prompt=caption_prompt,
            )
        elif self.sender_type == "ram":
            self.sender_ckb = SenderCKB_RAM(ram_ckpt=ram_ckpt, use_real_ckb=use_real_ckb, device=self.runtime_device)
        else:
            raise ValueError("sender_type must be 'blip' or 'ram'.")

        if use_real_receiver_ckb is None:
            use_real_receiver_ckb = bool(use_real_ckb)
        self.receiver_ckb = ReceiverCKB_SD(
            sd_dir=sd_dir,
            use_real_ckb=bool(use_real_receiver_ckb),
            device=self.runtime_device,
        )

        self.tokenizer = SimpleTextTokenizer(
            vocab_size=self.vocab_size,
            tokenizer_dir=blip_dir,
            use_hf_tokenizer=True,
            must_use_hf_tokenizer=True,
        )
        self.vocab_size = int(self.tokenizer.vocab_size)
        self.embedding = nn.Embedding(self.vocab_size, self.feature_dim)
        self.register_buffer(
            "pos_encoding",
            self._build_sinusoidal_positional_encoding(self.max_text_len, self.feature_dim),
            persistent=False,
        )

        self.semantic_encoder = SemanticEncoder(feature_dim=self.feature_dim, num_layers=3, num_heads=8, use_nam=self.use_nam)
        self.channel_encoder = ChannelEncoder(input_dim=self.feature_dim, output_dim=self.channel_dim, use_nam=self.use_nam)
        self.channel = PhysicalChannel(channel_type=channel_type)
        self.channel_decoder = ChannelDecoder(input_dim=self.channel_dim, output_dim=self.feature_dim, use_nam=self.use_nam)
        self.semantic_decoder = SemanticDecoder(feature_dim=self.feature_dim, num_layers=3, num_heads=8, use_nam=self.use_nam)
        self.lm_head = nn.Linear(self.feature_dim, self.vocab_size)

        self.enable_med = bool(enable_med)
        self.med = MED(**(med_kwargs or {})) if self.enable_med else None

    @staticmethod
    def _build_sinusoidal_positional_encoding(max_len: int, feature_dim: int) -> torch.Tensor:
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / feature_dim)
        )
        pe = torch.zeros(max_len, feature_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _add_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.max_text_len:
            raise RuntimeError(f"Sequence length {seq_len} exceeds max_text_len={self.max_text_len}")
        pos_enc: torch.Tensor = self.pos_encoding
        return x + pos_enc[:, :seq_len, :].to(device=x.device, dtype=x.dtype)

    @staticmethod
    def _build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def _prepare_decoder_inputs(self, target_ids: torch.Tensor) -> torch.Tensor:
        decoder_input_ids = target_ids.clone()
        decoder_input_ids[:, 1:] = target_ids[:, :-1]
        decoder_input_ids[:, 0] = int(self.tokenizer.bos_id)
        return decoder_input_ids

    @staticmethod
    def _assert_shape_dtype(name: str, x: torch.Tensor, expected_shape: tuple, expected_dtype: Optional[torch.dtype] = None) -> None:
        if tuple(x.shape) != tuple(expected_shape):
            raise RuntimeError(f"{name} shape mismatch: expected {expected_shape}, got {tuple(x.shape)}")
        if expected_dtype is not None and x.dtype != expected_dtype:
            raise RuntimeError(f"{name} dtype mismatch: expected {expected_dtype}, got {x.dtype}")

    def _encode_text(self, texts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        token_ids = self.tokenizer.encode(texts, max_len=self.max_text_len).to(device)
        attention_mask = (token_ids != self.tokenizer.pad_id).long()
        return {"token_ids": token_ids, "attention_mask": attention_mask}

    def _update_med(
        self,
        source_text: str,
        src_ids: torch.Tensor,
        semantic_seq: torch.Tensor,
        src_mask: torch.Tensor,
        image_id: str,
        dataset_id: str,
    ) -> Dict[str, int]:
        if self.med is None:
            return {"triggered": 0, "moved": 0, "stm_size": 0, "ltm_size": 0}

        denom = src_mask.sum(dim=1, keepdim=True).clamp(min=1)
        masked_x = semantic_seq * src_mask.unsqueeze(-1)
        med_feature = masked_x.sum(dim=1) / denom

        self.med.add_to_stm(
            {
                "image_id": image_id,
                "caption_text": source_text,
                "token_ids": src_ids[0],
                "semantic_feature": med_feature[0],
                "dataset_id": dataset_id,
            }
        )
        return self.med.maybe_update()

    def update_med_from_source_text(
        self,
        *,
        source_text: str,
        image_id: str,
        dataset_id: str,
    ) -> Dict[str, int]:
        if self.med is None:
            raise RuntimeError("MED is disabled; update_med_from_source_text is not allowed.")

        device = next(self.parameters()).device
        text_pack = self._encode_text([str(source_text)], device=device)
        src_ids = text_pack["token_ids"]
        src_mask = text_pack["attention_mask"]
        src_padding_mask = ~(src_mask.bool())

        src_embed = self._add_positional_encoding(self.embedding(src_ids))
        semantic_seq = self.semantic_encoder(src_embed, src_key_padding_mask=src_padding_mask, snr=0.0)

        denom = src_mask.sum(dim=1, keepdim=True).clamp(min=1)
        masked_x = semantic_seq * src_mask.unsqueeze(-1)
        med_feature = masked_x.sum(dim=1) / denom

        self.med.add_to_stm(
            {
                "image_id": str(image_id),
                "caption_text": str(source_text),
                "token_ids": src_ids[0],
                "semantic_feature": med_feature[0],
                "dataset_id": str(dataset_id),
            }
        )
        return self.med.maybe_update()

    def _build_semantic_sequence(
        self,
        source_text: str | List[str],
        snr_db: float,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(source_text, str):
            texts = [source_text]
        elif isinstance(source_text, list):
            if len(source_text) == 0:
                raise RuntimeError("source_text list cannot be empty")
            texts = [str(x) for x in source_text]
        else:
            raise RuntimeError(f"Unsupported source_text type: {type(source_text)}")

        text_pack = self._encode_text(texts, device=device)
        src_ids = text_pack["token_ids"]
        src_mask = text_pack["attention_mask"]
        src_padding_mask = ~(src_mask.bool())

        src_embed = self._add_positional_encoding(self.embedding(src_ids))
        semantic_seq = self.semantic_encoder(src_embed, src_key_padding_mask=src_padding_mask, snr=snr_db)
        self._assert_shape_dtype("semantic_seq", semantic_seq, (src_ids.size(0), self.max_text_len, self.feature_dim), src_embed.dtype)
        return {
            "src_ids": src_ids,
            "src_mask": src_mask,
            "src_padding_mask": src_padding_mask,
            "semantic_seq": semantic_seq,
        }

    def _transmit_sequence(self, semantic_seq: torch.Tensor, snr_db: float) -> Dict[str, torch.Tensor]:
        channel_symbols = self.channel_encoder(semantic_seq, snr=snr_db)
        self._assert_shape_dtype(
            "channel_symbols",
            channel_symbols,
            (semantic_seq.size(0), self.max_text_len, self.channel_dim),
            semantic_seq.dtype,
        )
        received_symbols = self.channel(channel_symbols, snr_db=snr_db, normalize_power=True)
        self._assert_shape_dtype(
            "received_symbols",
            received_symbols,
            (semantic_seq.size(0), self.max_text_len, self.channel_dim),
            semantic_seq.dtype,
        )
        recovered_sequence = self.channel_decoder(received_symbols, snr=snr_db)
        self._assert_shape_dtype(
            "recovered_sequence",
            recovered_sequence,
            (semantic_seq.size(0), self.max_text_len, self.feature_dim),
            semantic_seq.dtype,
        )
        return {
            "channel_symbols": channel_symbols,
            "received_symbols": received_symbols,
            "recovered_sequence": recovered_sequence,
        }

    def _decode_with_teacher(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        target_ids: torch.Tensor,
        snr_db: float,
        device: torch.device,
    ) -> Dict[str, torch.Tensor | bool]:
        decoder_input_ids = self._prepare_decoder_inputs(target_ids)
        tgt_embed = self._add_positional_encoding(self.embedding(decoder_input_ids))
        tgt_mask = (decoder_input_ids != self.tokenizer.pad_id).long()
        tgt_padding_mask = ~(tgt_mask.bool())
        causal_mask = self._build_causal_mask(decoder_input_ids.size(1), device=device)
        dec_out = self.semantic_decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            snr=snr_db,
        )
        return {
            "logits": self.lm_head(dec_out),
            "used_shift_right": True,
            "used_causal_mask": True,
        }

    def forward_channel_phase(
        self,
        snr_db: float,
        image: Image.Image | torch.Tensor | None = None,
        source_text: Optional[str | List[str]] = None,
    ) -> Dict[str, Any]:
        device = next(self.parameters()).device
        if source_text is None:
            if image is None:
                raise RuntimeError("forward_channel_phase requires either image or source_text")
            source_text = self.sender_ckb.forward(image)

        semantic_pack = self._build_semantic_sequence(source_text=source_text, snr_db=snr_db, device=device)
        tx_pack = self._transmit_sequence(semantic_seq=semantic_pack["semantic_seq"].detach(), snr_db=snr_db)
        return {
            "source_text": source_text,
            "semantic_seq": semantic_pack["semantic_seq"],
            "semantic_seq_detached": semantic_pack["semantic_seq"].detach(),
            "semantic_seq_teacher": semantic_pack["semantic_seq"].detach(),
            "channel_symbols": tx_pack["channel_symbols"],
            "received_symbols": tx_pack["received_symbols"],
            "recovered_sequence": tx_pack["recovered_sequence"],
            "recovered_seq": tx_pack["recovered_sequence"],
            "source_token_ids": semantic_pack["src_ids"],
            "source_attention_mask": semantic_pack["src_mask"],
            "padding_mask": semantic_pack["src_padding_mask"],
        }

    def forward_semantic_phase(
        self,
        snr_db: float,
        image: Image.Image | torch.Tensor | None = None,
        tgt_ids: Optional[torch.Tensor] = None,
        source_text: Optional[str | List[str]] = None,
    ) -> Dict[str, Any]:
        device = next(self.parameters()).device
        channel_phase = self.forward_channel_phase(image=image, snr_db=snr_db, source_text=source_text)
        src_ids: torch.Tensor = channel_phase["source_token_ids"]
        src_padding_mask = ~(channel_phase["source_attention_mask"].bool())
        if tgt_ids is None:
            tgt_ids = src_ids
        tgt_ids = tgt_ids.to(device)
        decode_pack = self._decode_with_teacher(
            memory=channel_phase["recovered_sequence"],
            memory_key_padding_mask=src_padding_mask,
            target_ids=tgt_ids,
            snr_db=snr_db,
            device=device,
        )
        channel_phase["logits"] = decode_pack["logits"]
        channel_phase["target_ids"] = tgt_ids
        channel_phase["used_shift_right"] = bool(decode_pack["used_shift_right"])
        channel_phase["used_causal_mask"] = bool(decode_pack["used_causal_mask"])
        return channel_phase

    def forward_joint_phase(
        self,
        snr_db: float,
        image: Image.Image | torch.Tensor | None = None,
        tgt_ids: Optional[torch.Tensor] = None,
        source_text: Optional[str | List[str]] = None,
        image_id: str = "sample_0",
        dataset_id: str = "default",
    ) -> Dict[str, Any]:
        out = self.forward_semantic_phase(
            image=image,
            snr_db=snr_db,
            tgt_ids=tgt_ids,
            source_text=source_text,
        )
        out["med_status"] = self.get_med_state()
        return out

    def get_med_state(self) -> Dict[str, int]:
        if self.med is None:
            return {"enabled": 0, "stm_size": 0, "ltm_size": 0}
        return {"enabled": 1, "stm_size": len(self.med.stm), "ltm_size": len(self.med.ltm)}

    def sample_med_batch(self, batch_size: int, stm_ratio: float = 0.5):
        if self.med is None:
            raise RuntimeError("MED is disabled; sample_med_batch is not allowed.")
        return self.med.sample_train_batch(batch_size=batch_size, stm_ratio=stm_ratio)

    def forward_text_train(
        self,
        image: Image.Image | torch.Tensor,
        snr_db: float,
        tgt_ids: Optional[torch.Tensor] = None,
        source_text: Optional[str] = None,
        image_id: str = "sample_0",
        dataset_id: str = "default",
    ) -> Dict[str, torch.Tensor | str]:
        out = self.forward_joint_phase(
            image=image,
            snr_db=snr_db,
            tgt_ids=tgt_ids,
            source_text=source_text,
            image_id=image_id,
            dataset_id=dataset_id,
        )
        return {
            "source_text": out["source_text"],
            "logits": out["logits"],
            "target_ids": out["target_ids"],
            "med_status": out["med_status"],
        }

    def _beam_search_decode(
        self,
        memory: torch.Tensor,
        src_padding_mask: torch.Tensor,
        snr_db: float,
        beam_size: int = 4,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Beam search decode (only supports B=1 inference)."""
        bos_id = int(self.tokenizer.bos_id)
        eos_id = int(self.tokenizer.eos_id)
        pad_id = int(self.tokenizer.pad_id)
        min_len = min(4, max(2, self.max_text_len))
        if device is None:
            device = next(self.parameters()).device

        exp_memory = memory.expand(beam_size, -1, -1).contiguous()
        exp_mask = src_padding_mask.expand(beam_size, -1).contiguous()

        seqs = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=device)
        scores = torch.full((beam_size,), float("-inf"), device=device)
        scores[0] = 0.0

        completed: list = []

        for step in range(self.max_text_len - 1):
            if len(completed) >= beam_size:
                break
            n_active = seqs.size(0)
            if n_active == 0:
                break

            tgt_embed = self._add_positional_encoding(self.embedding(seqs))
            tgt_pad_mask = ~(seqs != pad_id).bool()
            causal_mask = self._build_causal_mask(seqs.size(1), device=device)
            dec_out = self.semantic_decoder(
                tgt=tgt_embed,
                memory=exp_memory[:n_active],
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=exp_mask[:n_active],
                snr=snr_db,
            )
            logits = self.lm_head(dec_out)[:, -1, :].clone()
            logits[:, pad_id] = float("-inf")
            if step > 0:
                logits[:, bos_id] = float("-inf")
            if step < min_len - 1:
                logits[:, eos_id] = float("-inf")

            log_probs = F.log_softmax(logits, dim=-1)
            cand = scores[:n_active].unsqueeze(1) + log_probs
            cand_flat = cand.view(-1)
            vocab_size = log_probs.size(-1)

            k_expand = min(beam_size * 2, cand_flat.size(0))
            top_scores_t, top_ids_t = cand_flat.topk(k_expand)

            new_seqs: list = []
            new_scores: list = []

            for flat_id, sc in zip(top_ids_t.tolist(), top_scores_t.tolist()):
                beam_id = flat_id // vocab_size
                token_id = flat_id % vocab_size
                new_seq = torch.cat([
                    seqs[beam_id],
                    torch.tensor([token_id], dtype=torch.long, device=device),
                ])
                if token_id == eos_id:
                    length_norm = new_seq.size(0) ** 0.6
                    completed.append((sc / length_norm, new_seq))
                else:
                    if len(new_seqs) < beam_size:
                        new_seqs.append(new_seq)
                        new_scores.append(sc)
                if len(new_seqs) >= beam_size and len(completed) >= beam_size:
                    break

            if not new_seqs:
                break

            max_slen = max(s.size(0) for s in new_seqs)
            n_new = len(new_seqs)
            seqs = torch.full((n_new, max_slen), pad_id, dtype=torch.long, device=device)
            for i, s in enumerate(new_seqs):
                seqs[i, :s.size(0)] = s
            scores = torch.tensor(new_scores, dtype=torch.float, device=device)

            exp_memory = memory.expand(n_new, -1, -1).contiguous()
            exp_mask = src_padding_mask.expand(n_new, -1).contiguous()

        if not completed:
            for i in range(seqs.size(0)):
                sc = scores[i].item()
                if sc != float("-inf"):
                    length_norm = max(seqs[i].size(0), 1) ** 0.6
                    completed.append((sc / length_norm, seqs[i]))
        if not completed:
            return torch.full((1, 1), bos_id, dtype=torch.long, device=device)

        best_seq = max(completed, key=lambda x: x[0])[1]
        return best_seq.unsqueeze(0)

    @torch.inference_mode()
    def infer_full(
        self,
        image: Image.Image | torch.Tensor,
        snr_db: float,
        sd_height: int = 512,
        sd_width: int = 512,
        sd_steps: int = 30,
        sd_guidance: float = 7.5,
        sd_seed: int = 42,
        return_debug: bool = True,
        decode_strategy: str = "beam",
        beam_size: int = 4,
    ) -> Dict[str, object]:
        """Full inference pipeline: image -> channel -> reconstructed image."""
        device = next(self.parameters()).device
        source_text = self.sender_ckb.forward(image)
        channel_phase = self.forward_channel_phase(image=image, snr_db=snr_db, source_text=source_text)
        src_ids = channel_phase["source_token_ids"]
        src_padding_mask = ~(channel_phase["source_attention_mask"].bool())
        memory = channel_phase["recovered_sequence"]
        channel_symbols = channel_phase["channel_symbols"]
        received_symbols = channel_phase["received_symbols"]
        recovered_sequence = channel_phase["recovered_sequence"]

        B = src_ids.size(0)
        if decode_strategy == "beam" and B == 1:
            recovered_ids = self._beam_search_decode(
                memory=memory,
                src_padding_mask=src_padding_mask,
                snr_db=snr_db,
                beam_size=beam_size,
                device=device,
            )
        else:
            generated_ids = torch.full(
                (B, 1),
                int(self.tokenizer.bos_id),
                dtype=torch.long,
                device=device,
            )
            eos_id = int(self.tokenizer.eos_id)
            min_generation_len = min(4, max(2, self.max_text_len))

            for _ in range(self.max_text_len - 1):
                tgt_embed = self._add_positional_encoding(self.embedding(generated_ids))
                tgt_pad_mask = ~(generated_ids != self.tokenizer.pad_id).bool()
                causal_mask = self._build_causal_mask(generated_ids.size(1), device=device)
                dec_out = self.semantic_decoder(
                    tgt=tgt_embed,
                    memory=memory,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                    memory_key_padding_mask=src_padding_mask,
                    snr=snr_db,
                )
                logits = self.lm_head(dec_out)
                logits = logits.clone()
                logits[:, -1, int(self.tokenizer.pad_id)] = float("-inf")
                if generated_ids.size(1) > 1:
                    logits[:, -1, int(self.tokenizer.bos_id)] = float("-inf")
                if generated_ids.size(1) < min_generation_len:
                    logits[:, -1, eos_id] = float("-inf")
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_id], dim=1)

                if bool((next_id == eos_id).all()):
                    break

            recovered_ids = generated_ids

        recovered_text = self.tokenizer.decode(recovered_ids)[0]

        if self.receiver_ckb.use_real_ckb:
            reconstructed_image = self.receiver_ckb.forward(
                recovered_text,
                height=sd_height,
                width=sd_width,
                num_inference_steps=sd_steps,
                guidance_scale=sd_guidance,
                seed=sd_seed,
            )
        else:
            reconstructed_image = None

        result = {
            "source_text": source_text,
            "recovered_text": recovered_text,
            "reconstructed_image": reconstructed_image,
            "token_ids": recovered_ids,
            "source_token_ids": src_ids,
            "generated_ids": recovered_ids,
            "channel_symbols": channel_symbols,
            "received_symbols": received_symbols,
            "recovered_sequence": recovered_sequence,
        }

        if not return_debug:
            return {
                "source_text": result["source_text"],
                "recovered_text": result["recovered_text"],
                "reconstructed_image": result["reconstructed_image"],
                "token_ids": result["token_ids"],
            }
        return result
