"""Step 5 smoke test: single-batch forward and one-epoch training checks."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from channels.awgn import awgn_channel
from models.channel_codec import ChannelDecoder, ChannelEncoder
from models.kb_blip import BlipKnowledgeBase
from models.kb_sd import StableDiffusionKnowledgeBase
from models.semantic_codec import SemanticDecoder, SemanticEncoder
from trainers.trainer_channel import ChannelTrainer
from trainers.trainer_joint import JointTrainConfig, JointTrainer
from trainers.trainer_semantic import SemanticTrainer
from utils.seed import set_seed


def _toy_batches(num_batches: int = 2, batch_size: int = 2, seq_len: int = 12, vocab_size: int = 500):
    batches = []
    for _ in range(num_batches):
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        snr = torch.ones(batch_size, 1)
        batches.append({"token_ids": token_ids, "attention_mask": attention_mask, "snr": snr, "snr_db": 4.0})
    return batches


def run_step5_smoke(seed: int = 42) -> dict:
    set_seed(seed)
    device = "cpu"

    blip = BlipKnowledgeBase(
        checkpoint="Salesforce/blip-image-captioning-base",
        max_length=32,
        device=device,
        load_pretrained=False,
        allow_fallback=True,
    )
    sd = StableDiffusionKnowledgeBase(
        checkpoint="sd-legacy/stable-diffusion-v1-5",
        image_size=64,
        seed=seed,
        device=device,
        load_pretrained=False,
        allow_fallback=True,
    )

    semantic_encoder = SemanticEncoder(vocab_size=500, d_model=128, n_heads=8, num_layers=3).to(device)
    semantic_decoder = SemanticDecoder(vocab_size=500, d_model=128, n_heads=8, num_layers=3).to(device)
    channel_encoder = ChannelEncoder(d_model=128, hidden1=256, hidden2=128, symbol_dim=128).to(device)
    channel_decoder = ChannelDecoder(d_model=128, hidden1=256, hidden2=128, symbol_dim=128).to(device)

    images = torch.rand(2, 3, 224, 224, device=device)
    captions = blip.generate_caption(images)
    tokens = blip.tokenize(captions)
    token_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    snr = torch.ones(token_ids.shape[0], 1, device=device)

    semantic_features = semantic_encoder(token_ids=token_ids, snr=snr)
    symbols = channel_encoder(semantic_features=semantic_features, snr=snr)
    received = awgn_channel(symbols, snr_db=4.0, training_mode=False, seed=seed)
    decoded_features = channel_decoder(received_symbols=received, snr=snr)
    logits = semantic_decoder(channel_features=decoded_features, target_ids=token_ids, snr=snr)
    decoded_ids = semantic_decoder.greedy_decode(
        channel_features=decoded_features,
        bos_id=1,
        eos_id=2,
        max_len=token_ids.shape[1],
    )
    recon_images = sd.reconstruct_from_text(captions)

    toy_loader = _toy_batches()
    trainer_a = ChannelTrainer(
        semantic_encoder=semantic_encoder,
        channel_encoder=channel_encoder,
        channel_decoder=channel_decoder,
        dataloader=toy_loader,
        channel_fn=awgn_channel,
        device=device,
        lr=1e-4,
        seed=seed,
    )
    trainer_b = SemanticTrainer(
        semantic_encoder=semantic_encoder,
        semantic_decoder=semantic_decoder,
        channel_encoder=channel_encoder,
        channel_decoder=channel_decoder,
        dataloader=toy_loader,
        channel_fn=awgn_channel,
        device=device,
        lr=1e-4,
        seed=seed,
    )

    stage_a = trainer_a.train_one_epoch()
    stage_b = trainer_b.train_one_epoch()
    joint = JointTrainer(trainer_a, trainer_b, config=JointTrainConfig(max_rounds=2, patience=1))
    stage_c = joint.fit()

    result = {
        "single_batch": {
            "captions": len(captions),
            "token_ids_shape": list(token_ids.shape),
            "attention_mask_shape": list(attention_mask.shape),
            "semantic_features_shape": list(semantic_features.shape),
            "symbols_shape": list(symbols.shape),
            "decoded_features_shape": list(decoded_features.shape),
            "logits_shape": list(logits.shape),
            "decoded_ids_shape": list(decoded_ids.shape),
            "recon_images_shape": list(recon_images.shape),
        },
        "stage_a": stage_a,
        "stage_b": stage_b,
        "stage_c": {
            "best_round": stage_c["best_round"],
            "best_combined_val_loss": stage_c["best_combined_val_loss"],
            "history_len": len(stage_c["history"]),
        },
    }
    return result


def main() -> None:
    print("SMOKE MODE / NOT FORMAL EXPERIMENT")
    result = run_step5_smoke(seed=42)
    out_dir = Path("outputs") / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "step5_smoke_result.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STEP5_SMOKE_OK")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
