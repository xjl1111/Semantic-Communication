"""Train-all entrypoint with staged A/B/C orchestration.

Default mode is formal: use prepared text cache and run Stage A/B/C with no proxy fallback path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from channels.awgn import awgn_channel
from channels.rayleigh import rayleigh_channel
from data.cache import read_caption_cache
from models.channel_codec import ChannelDecoder, ChannelEncoder
from models.semantic_codec import SemanticDecoder, SemanticEncoder
from scripts.smoke_test import run_step5_smoke
from trainers.trainer_channel import ChannelTrainer
from trainers.trainer_joint import JointTrainConfig, JointTrainer
from trainers.trainer_semantic import SemanticTrainer
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["formal", "smoke"], default="formal")
    parser.add_argument("--stage", choices=["all", "a", "b", "c"], default="all")
    parser.add_argument("--cache-file", type=str, default="outputs/cache/captions/cifar_train.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/train_all")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--semantic-layers", type=int, default=3)
    parser.add_argument("--channel-hidden1", type=int, default=256)
    parser.add_argument("--channel-hidden2", type=int, default=128)
    parser.add_argument("--symbol-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs-a", type=int, default=10)
    parser.add_argument("--epochs-b", type=int, default=10)
    parser.add_argument("--joint-rounds", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn")
    parser.add_argument("--snr-mode", choices=["fixed", "uniform"], default="fixed")
    parser.add_argument("--snr-db", type=float, default=4.0)
    parser.add_argument("--snr-min-db", type=float, default=0.0)
    parser.add_argument("--snr-max-db", type=float, default=10.0)
    return parser.parse_args()


def _pick_channel_fn(channel_name: str):
    if channel_name == "rayleigh":
        return rayleigh_channel
    return awgn_channel


def _build_batches_from_cache(
    cache_file: Path,
    batch_size: int,
    max_len: int,
    snr_mode: str,
    snr_db: float,
    snr_min_db: float,
    snr_max_db: float,
    seed: int,
) -> tuple[list[dict], int]:
    records = read_caption_cache(cache_file)
    if not records:
        raise FileNotFoundError(f"No records found in cache: {cache_file}")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    token_tensors: List[torch.Tensor] = []
    attn_tensors: List[torch.Tensor] = []
    max_token_id = 0

    for item in records:
        ids = [int(v) for v in item.tokenizer_ids if isinstance(v, (int, float))]
        if len(ids) < 2:
            continue
        ids = ids[:max_len]
        max_token_id = max(max_token_id, max(ids))

        tok = torch.tensor(ids, dtype=torch.long)
        attn = torch.ones_like(tok)
        if tok.shape[0] < max_len:
            pad_len = max_len - tok.shape[0]
            tok = torch.cat([tok, torch.zeros(pad_len, dtype=torch.long)], dim=0)
            attn = torch.cat([attn, torch.zeros(pad_len, dtype=torch.long)], dim=0)

        token_tensors.append(tok)
        attn_tensors.append(attn)

    if not token_tensors:
        raise ValueError(f"No valid tokenizer_ids with length>=2 found in {cache_file}")

    token_stack = torch.stack(token_tensors, dim=0)
    attn_stack = torch.stack(attn_tensors, dim=0)

    batches: list[dict] = []
    total = token_stack.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        token_batch = token_stack[start:end]
        attn_batch = attn_stack[start:end]
        cur_bsz = token_batch.shape[0]

        if snr_mode == "uniform":
            snr_vec = torch.rand((cur_bsz, 1), generator=rng) * (snr_max_db - snr_min_db) + snr_min_db
            snr_db_batch = float(torch.mean(snr_vec).item())
        else:
            snr_vec = torch.full((cur_bsz, 1), fill_value=snr_db, dtype=torch.float32)
            snr_db_batch = float(snr_db)

        batches.append(
            {
                "token_ids": token_batch,
                "attention_mask": attn_batch,
                "snr": snr_vec,
                "snr_db": snr_db_batch,
            }
        )

    return batches, max_token_id + 1


def _ensure_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _run_formal(args: argparse.Namespace) -> dict:
    cache_file = Path(args.cache_file)
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Cache file not found: {cache_file}. Run scripts/prepare_text_cache.py first."
        )

    device = _ensure_device(args.device)
    seeds = args.seeds if args.seeds else [args.seed]

    base_dataloader, vocab_size_from_cache = _build_batches_from_cache(
        cache_file=cache_file,
        batch_size=args.batch_size,
        max_len=args.max_len,
        snr_mode=args.snr_mode,
        snr_db=args.snr_db,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
        seed=seeds[0],
    )
    vocab_size = max(args.vocab_size, vocab_size_from_cache, 128)

    result: dict = {
        "mode": "formal",
        "stage": args.stage,
        "cache_file": str(cache_file),
        "device": device,
        "num_batches": len(base_dataloader),
        "vocab_size": vocab_size,
        "channel": args.channel,
        "symbol_dim": args.symbol_dim,
        "seeds": seeds,
    }

    per_seed: list[Dict[str, object]] = []
    channel_fn = _pick_channel_fn(args.channel)

    for seed in seeds:
        set_seed(seed)
        dataloader, _ = _build_batches_from_cache(
            cache_file=cache_file,
            batch_size=args.batch_size,
            max_len=args.max_len,
            snr_mode=args.snr_mode,
            snr_db=args.snr_db,
            snr_min_db=args.snr_min_db,
            snr_max_db=args.snr_max_db,
            seed=seed,
        )

        semantic_encoder = SemanticEncoder(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            num_layers=args.semantic_layers,
        ).to(device)
        semantic_decoder = SemanticDecoder(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            num_layers=args.semantic_layers,
        ).to(device)
        channel_encoder = ChannelEncoder(
            d_model=args.d_model,
            hidden1=args.channel_hidden1,
            hidden2=args.channel_hidden2,
            symbol_dim=args.symbol_dim,
        ).to(device)
        channel_decoder = ChannelDecoder(
            d_model=args.d_model,
            hidden1=args.channel_hidden1,
            hidden2=args.channel_hidden2,
            symbol_dim=args.symbol_dim,
        ).to(device)

        trainer_a = ChannelTrainer(
            semantic_encoder=semantic_encoder,
            channel_encoder=channel_encoder,
            channel_decoder=channel_decoder,
            dataloader=dataloader,
            channel_fn=channel_fn,
            device=device,
            lr=args.lr,
            seed=seed,
        )
        trainer_b = SemanticTrainer(
            semantic_encoder=semantic_encoder,
            semantic_decoder=semantic_decoder,
            channel_encoder=channel_encoder,
            channel_decoder=channel_decoder,
            dataloader=dataloader,
            channel_fn=channel_fn,
            device=device,
            lr=args.lr,
            seed=seed,
        )

        seed_result: Dict[str, object] = {"seed": seed}
        if args.stage in ("all", "a"):
            hist_a = []
            for _ in range(args.epochs_a):
                train_stats = trainer_a.train_one_epoch()
                val_stats = trainer_a.validate()
                hist_a.append({"train": train_stats, "val": val_stats})
            seed_result["stage_a"] = hist_a

        if args.stage in ("all", "b"):
            hist_b = []
            for _ in range(args.epochs_b):
                train_stats = trainer_b.train_one_epoch()
                val_stats = trainer_b.validate()
                hist_b.append({"train": train_stats, "val": val_stats})
            seed_result["stage_b"] = hist_b

        if args.stage in ("all", "c"):
            joint = JointTrainer(
                trainer_a,
                trainer_b,
                config=JointTrainConfig(max_rounds=args.joint_rounds, patience=args.patience),
            )
            seed_result["stage_c"] = joint.fit()

        per_seed.append(seed_result)

    result["per_seed"] = per_seed
    best_losses = [
        float(item["stage_c"].get("best_combined_val_loss", math.inf))
        for item in per_seed
        if "stage_c" in item
    ]
    if best_losses:
        result["summary"] = {
            "best_combined_val_loss_mean": float(sum(best_losses) / len(best_losses)),
            "best_combined_val_loss_values": best_losses,
        }
    else:
        result["summary"] = {"best_combined_val_loss_mean": math.inf, "best_combined_val_loss_values": []}
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "train_all_result.json"

    if args.mode == "smoke":
        print("SMOKE MODE / NOT FORMAL EXPERIMENT")
        result = run_step5_smoke(seed=args.seed)
        payload = {"mode": "smoke", "result": result}
    else:
        print("FORMAL MODE")
        payload = _run_formal(args)

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TRAIN_ALL_DONE")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
