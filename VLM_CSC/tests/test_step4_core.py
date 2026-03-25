"""Step 4 smoke tests for SD wrapper, losses, and trainers."""

import torch

from channels.awgn import awgn_channel
from losses.mutual_info_proxy import channel_proxy_loss
from losses.text_ce import text_cross_entropy
from models.channel_codec import ChannelDecoder, ChannelEncoder
from models.kb_sd import StableDiffusionKnowledgeBase
from models.semantic_codec import SemanticDecoder, SemanticEncoder
from trainers.trainer_channel import ChannelTrainer
from trainers.trainer_joint import JointTrainConfig, JointTrainer
from trainers.trainer_semantic import SemanticTrainer


def _toy_dataloader(num_batches: int = 2, batch_size: int = 2, seq_len: int = 8, vocab_size: int = 200):
    batches = []
    for _ in range(num_batches):
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attn = torch.ones(batch_size, seq_len)
        snr = torch.ones(batch_size, 1)
        batches.append({"token_ids": token_ids, "attention_mask": attn, "snr": snr, "snr_db": 4.0})
    return batches


def test_sd_wrapper_fallback_shape() -> None:
    sd = StableDiffusionKnowledgeBase(
        checkpoint="sd-legacy/stable-diffusion-v1-5",
        image_size=64,
        seed=7,
        load_pretrained=False,
    )
    images = sd.reconstruct_from_text(["a cat", "a dog"])
    assert images.shape == (2, 3, 64, 64)


def test_losses_finite() -> None:
    logits = torch.randn(2, 6, 50)
    targets = torch.randint(0, 50, (2, 6))
    mask = torch.ones(2, 6)
    loss_ce = text_cross_entropy(logits, targets, mask)

    decoded = torch.randn(2, 6, 128)
    original = torch.randn(2, 6, 128)
    symbols = torch.randn(2, 6, 128)
    loss_proxy = channel_proxy_loss(decoded, original, symbols)

    assert torch.isfinite(loss_ce)
    assert torch.isfinite(loss_proxy)


def test_trainers_smoke() -> None:
    vocab_size = 200
    semantic_encoder = SemanticEncoder(vocab_size=vocab_size)
    semantic_decoder = SemanticDecoder(vocab_size=vocab_size)
    channel_encoder = ChannelEncoder()
    channel_decoder = ChannelDecoder()

    loader = _toy_dataloader(vocab_size=vocab_size)

    trainer_a = ChannelTrainer(
        semantic_encoder=semantic_encoder,
        channel_encoder=channel_encoder,
        channel_decoder=channel_decoder,
        dataloader=loader,
        channel_fn=awgn_channel,
        device="cpu",
    )
    a_stats = trainer_a.train_one_epoch()
    assert "loss" in a_stats and a_stats["steps"] > 0

    trainer_b = SemanticTrainer(
        semantic_encoder=semantic_encoder,
        semantic_decoder=semantic_decoder,
        channel_encoder=channel_encoder,
        channel_decoder=channel_decoder,
        dataloader=loader,
        channel_fn=awgn_channel,
        device="cpu",
    )
    b_stats = trainer_b.train_one_epoch()
    assert "loss" in b_stats and b_stats["steps"] > 0

    joint = JointTrainer(trainer_a, trainer_b, config=JointTrainConfig(max_rounds=2, patience=1))
    result = joint.fit()
    assert "best_round" in result
    assert len(result["history"]) >= 1
