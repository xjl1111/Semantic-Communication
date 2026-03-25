"""Step 3 smoke tests for core sender/channel modules."""

import torch

from channels.awgn import awgn_channel
from models.channel_codec import ChannelDecoder, ChannelEncoder
from models.kb_blip import BlipKnowledgeBase
from models.semantic_codec import SemanticDecoder, SemanticEncoder


def test_blip_wrapper_interface() -> None:
    kb = BlipKnowledgeBase(
        checkpoint="Salesforce/blip-image-captioning-base",
        max_length=16,
        load_pretrained=False,
    )
    images = torch.rand(2, 3, 224, 224)
    captions = kb.generate_caption(images)
    tokens = kb.tokenize(captions)

    assert len(captions) == 2
    assert tokens["input_ids"].shape == (2, 16)
    assert tokens["attention_mask"].shape == (2, 16)


def test_semantic_encoder_decoder_shapes() -> None:
    encoder = SemanticEncoder(vocab_size=500, d_model=128, n_heads=8, num_layers=3)
    decoder = SemanticDecoder(vocab_size=500, d_model=128, n_heads=8, num_layers=3)

    token_ids = torch.randint(low=0, high=499, size=(2, 12))
    snr = torch.randn(2, 1)
    encoded = encoder(token_ids=token_ids, snr=snr)
    logits = decoder(channel_features=encoded, target_ids=token_ids, snr=snr)

    assert encoded.shape == (2, 12, 128)
    assert logits.shape == (2, 12, 500)


def test_channel_codec_shapes() -> None:
    channel_encoder = ChannelEncoder(d_model=128, hidden1=256, hidden2=128, symbol_dim=128)
    channel_decoder = ChannelDecoder(d_model=128, hidden1=256, hidden2=128, symbol_dim=128)

    x = torch.randn(2, 10, 128)
    snr = torch.ones(2, 1)

    symbols = channel_encoder(semantic_features=x, snr=snr)
    noisy = awgn_channel(symbols, snr_db=5.0, training_mode=False, seed=13)
    decoded = channel_decoder(received_symbols=noisy, snr=snr)

    assert symbols.shape == (2, 10, 128)
    assert decoded.shape == (2, 10, 128)
