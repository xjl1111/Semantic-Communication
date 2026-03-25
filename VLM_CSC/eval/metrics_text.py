"""Text metrics: BLEU-1/BLEU-2 and utilities."""

import math
from collections import Counter
from typing import Iterable, Sequence


def bleu_score(preds: Sequence[str], refs: Sequence[str], n: int = 4) -> float:
    """Compute corpus BLEU-n with clipped precision and brevity penalty."""
    if len(preds) != len(refs):
        raise ValueError("preds and refs must have the same length")
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(preds) == 0:
        return 0.0

    total_clipped = 0
    total_count = 0
    pred_len = 0
    ref_len = 0

    for pred, ref in zip(preds, refs):
        p_tokens = pred.strip().split()
        r_tokens = ref.strip().split()
        pred_len += len(p_tokens)
        ref_len += len(r_tokens)

        if len(p_tokens) < n:
            continue

        pred_ngrams = Counter(tuple(p_tokens[i : i + n]) for i in range(len(p_tokens) - n + 1))
        ref_ngrams = Counter(tuple(r_tokens[i : i + n]) for i in range(max(0, len(r_tokens) - n + 1)))

        clipped = 0
        for ng, count in pred_ngrams.items():
            clipped += min(count, ref_ngrams.get(ng, 0))
        total_clipped += clipped
        total_count += sum(pred_ngrams.values())

    if total_count == 0:
        return 0.0
    precision = total_clipped / total_count
    if precision <= 0:
        return 0.0

    if pred_len == 0:
        return 0.0
    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1 - (ref_len / pred_len))
    return float(brevity_penalty * precision)


def corpus_bleu_12(preds: Iterable[str], refs: Iterable[str]) -> dict:
    """Return BLEU-1 and BLEU-2."""
    pred_list = list(preds)
    ref_list = list(refs)
    return {
        "bleu1": bleu_score(pred_list, ref_list, n=1),
        "bleu2": bleu_score(pred_list, ref_list, n=2),
    }
