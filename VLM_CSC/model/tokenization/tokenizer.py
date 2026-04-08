"""Text tokenization module."""
from __future__ import annotations

from typing import List

import torch


class SimpleTextTokenizer:
    """HuggingFace tokenizer wrapper with fallback support."""

    def __init__(
        self,
        vocab_size: int = 30522,
        pad_id: int = 0,
        bos_id: int = 101,
        eos_id: int = 102,
        tokenizer_dir: str | None = None,
        use_hf_tokenizer: bool = True,
        must_use_hf_tokenizer: bool = True,
    ):
        self.hf_tokenizer = None

        if use_hf_tokenizer:
            try:
                from transformers import AutoTokenizer

                if tokenizer_dir is not None:
                    self.hf_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
                else:
                    self.hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            except Exception as exc:
                if must_use_hf_tokenizer:
                    raise RuntimeError(f"Failed to load HF tokenizer: {exc}") from exc
                self.hf_tokenizer = None

        if must_use_hf_tokenizer and self.hf_tokenizer is None:
            raise RuntimeError("HF tokenizer is required but not available.")

        if self.hf_tokenizer is not None:
            self.vocab_size = int(self.hf_tokenizer.vocab_size)
            self.pad_id = int(self.hf_tokenizer.pad_token_id if self.hf_tokenizer.pad_token_id is not None else 0)

            if self.hf_tokenizer.bos_token_id is not None:
                self.bos_id = int(self.hf_tokenizer.bos_token_id)
            elif self.hf_tokenizer.cls_token_id is not None:
                self.bos_id = int(self.hf_tokenizer.cls_token_id)
            else:
                self.bos_id = int(bos_id)

            if self.hf_tokenizer.eos_token_id is not None:
                self.eos_id = int(self.hf_tokenizer.eos_token_id)
            elif self.hf_tokenizer.sep_token_id is not None:
                self.eos_id = int(self.hf_tokenizer.sep_token_id)
            else:
                self.eos_id = int(eos_id)
        else:
            self.vocab_size = int(vocab_size)
            self.pad_id = int(pad_id)
            self.bos_id = int(bos_id)
            self.eos_id = int(eos_id)

        tokenizer_name = self.hf_tokenizer.__class__.__name__ if self.hf_tokenizer is not None else "SimpleTextTokenizerFallback"
        print(
            f"[TOKENIZER] class={tokenizer_name}, vocab_size={self.vocab_size}, "
            f"bos_id={self.bos_id}, eos_id={self.eos_id}, pad_id={self.pad_id}"
        )

    def encode(self, texts: List[str], max_len: int = 32) -> torch.Tensor:
        if self.hf_tokenizer is not None:
            encoded = self.hf_tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            return encoded["input_ids"].long()

        batch_ids = []
        for text in texts:
            core = [(ord(ch) % (self.vocab_size - 103)) + 103 for ch in text][: max_len - 2]
            ids = [self.bos_id] + core + [self.eos_id]
            ids = ids + [self.pad_id] * (max_len - len(ids))
            batch_ids.append(ids)
        return torch.tensor(batch_ids, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> List[str]:
        if self.hf_tokenizer is not None:
            return self.hf_tokenizer.batch_decode(token_ids, skip_special_tokens=True)

        texts: List[str] = []
        for row in token_ids:
            chars: List[str] = []
            for token in row.tolist():
                if token in (self.pad_id, self.bos_id, self.eos_id):
                    continue
                val = max(32, min(126, (token - 103) % 95 + 32))
                chars.append(chr(val))
            texts.append("".join(chars).strip() if chars else "")
        return texts
