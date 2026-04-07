from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from PIL import Image

from communication_modules import NAM
from communication_modules import PhysicalChannel


def _load_vlm_module(vlm_file: Path):
    spec = importlib.util.spec_from_file_location("vlm_csc_module", str(vlm_file))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load VLM-CSC.py module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_channel_numerics() -> None:
    x = torch.randn(4, 128)

    awgn = PhysicalChannel(channel_type="awgn")
    y_awgn = awgn(x, snr_db=5.0)
    if y_awgn.shape != x.shape:
        raise RuntimeError("AWGN channel output shape mismatch.")
    if not torch.isfinite(y_awgn).all():
        raise RuntimeError("AWGN channel output contains non-finite values.")

    rayleigh = PhysicalChannel(channel_type="rayleigh", rayleigh_mode="fast")
    y_rayleigh = rayleigh(x, snr_db=5.0)
    if y_rayleigh.shape != x.shape:
        raise RuntimeError("Rayleigh channel output shape mismatch.")
    if not torch.isfinite(y_rayleigh).all():
        raise RuntimeError("Rayleigh channel output contains non-finite values.")

    print("[CHECK] channel numerics: OK")


def check_ckb_assets(base_dir: Path) -> None:
    models_dir = base_dir.parent / "data" / "assets" / "downloaded_models"
    blip_dir = models_dir / "blip"
    sd_dir = models_dir / "sd15"

    if not blip_dir.exists():
        raise RuntimeError("BLIP local CKB folder missing.")
    if not sd_dir.exists():
        raise RuntimeError("SD local CKB folder missing.")

    required_blip = ["config.json", "model.safetensors", "tokenizer.json"]
    for name in required_blip:
        if not (blip_dir / name).exists():
            raise RuntimeError(f"BLIP file missing: {name}")

    required_sd = ["model_index.json", "text_encoder", "unet", "vae"]
    for name in required_sd:
        if not (sd_dir / name).exists():
            raise RuntimeError(f"SD asset missing: {name}")

    print("[CHECK] CKB assets: OK")


def check_system_forward(base_dir: Path) -> None:
    vlm_module = _load_vlm_module(base_dir / "VLM-CSC.py")
    VLMCscSystem = vlm_module.VLMCscSystem

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VLMCscSystem(
        feature_dim=128,
        max_text_len=24,
        channel_type="awgn",
        use_real_ckb=False,
        enable_med=True,
        med_kwargs={
            "stm_max_size": 3,
            "tau": 10.0,
            "threshold": 0.05,
            "transfer_if": "greater",
        },
        blip_dir=base_dir.parent / "data" / "assets" / "downloaded_models" / "blip",
        sd_dir=base_dir.parent / "data" / "assets" / "downloaded_models" / "sd15",
        device=device,
    ).to(device)

    image = Image.new("RGB", (224, 224), color=(110, 130, 170))

    med_trace = []
    for idx in range(4):
        out = model.forward_text_train(
            image=image,
            snr_db=5.0,
            image_id=f"img_{idx}",
            dataset_id="smoke",
        )
        logits = out["logits"]
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("Train forward logits type invalid.")
        if logits.shape[:2] != (1, 24):
            raise RuntimeError(f"Train forward logits shape invalid: {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise RuntimeError("Train forward logits contain non-finite values.")
        med_trace.append(out["med_status"])

    infer_out = model.infer_full(image=image, snr_db=5.0)
    recovered_text = infer_out["recovered_text"]
    recon_image = infer_out["reconstructed_image"]

    if not isinstance(recovered_text, str):
        raise RuntimeError("Recovered text type invalid.")
    if len(recovered_text) == 0:
        raise RuntimeError("Recovered text is empty.")
    if not isinstance(recon_image, Image.Image):
        raise RuntimeError("Reconstructed image type invalid.")

    med_state = model.get_med_state()
    if med_state["ltm_size"] <= 0:
        raise RuntimeError("MED transfer did not trigger as expected.")

    sampled = model.sample_med_batch(batch_size=2, stm_ratio=0.5)
    if len(sampled) == 0:
        raise RuntimeError("MED sample_train_batch returned empty after updates.")

    source_text = infer_out["source_text"]
    if source_text == recovered_text:
        raise RuntimeError("Leakage suspicion: recovered_text exactly equals source_text before training.")

    src_ids = infer_out["source_token_ids"]
    generated_ids = infer_out["generated_ids"]
    channel_symbols = infer_out["channel_symbols"]
    received_symbols = infer_out["received_symbols"]
    recovered_pooled = infer_out["recovered_pooled"]

    if src_ids.shape != (1, 24):
        raise RuntimeError(f"source_token_ids shape mismatch: {tuple(src_ids.shape)}")
    if generated_ids.dim() != 2:
        raise RuntimeError("generated_ids must be 2D tensor.")
    if channel_symbols.shape != (1, 128) or received_symbols.shape != (1, 128) or recovered_pooled.shape != (1, 128):
        raise RuntimeError("sentence-level channel tensors shape mismatch.")
    if channel_symbols.dtype != received_symbols.dtype or channel_symbols.dtype != recovered_pooled.dtype:
        raise RuntimeError("channel path dtype mismatch.")

    print("[CHECK] system forward: OK")
    print("[CHECK] recovered_text length:", len(recovered_text))
    print("[CHECK] MED trace:", med_trace)
    print("[CHECK] MED state:", med_state)


def check_real_ckb_inference(base_dir: Path) -> None:
    vlm_module = _load_vlm_module(base_dir / "VLM-CSC.py")
    SenderCKB_BLIP = vlm_module.SenderCKB_BLIP
    ReceiverCKB_SD = vlm_module.ReceiverCKB_SD

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.new("RGB", (224, 224), color=(90, 140, 200))

    sender = SenderCKB_BLIP(base_dir.parent / "data" / "assets" / "downloaded_models" / "blip", use_real_ckb=True, device=device)
    caption = sender.forward(image)
    if not isinstance(caption, str) or len(caption.strip()) == 0:
        raise RuntimeError("Real BLIP inference failed to produce a caption.")
    word_count = len(caption.strip().split())
    if word_count < 3 or word_count > 30:
        raise RuntimeError(f"Real BLIP caption length out of range: {word_count}")

    receiver = ReceiverCKB_SD(base_dir.parent / "data" / "assets" / "downloaded_models" / "sd15", use_real_ckb=True, device=device)
    recon = receiver.forward(
        caption,
        height=256,
        width=256,
        num_inference_steps=1,
        guidance_scale=7.5,
    )
    if not isinstance(recon, Image.Image):
        raise RuntimeError("Real SD inference failed to produce an image.")
    if recon.size != (256, 256):
        raise RuntimeError(f"Real SD image size mismatch: {recon.size}")

    print("[CHECK] real CKB inference: OK")
    print("[CHECK] real BLIP caption:", caption)


def check_system_forward_real_ckb(base_dir: Path) -> None:
    vlm_module = _load_vlm_module(base_dir / "VLM-CSC.py")
    VLMCscSystem = vlm_module.VLMCscSystem

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VLMCscSystem(
        feature_dim=128,
        max_text_len=24,
        channel_type="awgn",
        use_real_ckb=True,
        enable_med=False,
        blip_dir=base_dir.parent / "data" / "assets" / "downloaded_models" / "blip",
        sd_dir=base_dir.parent / "data" / "assets" / "downloaded_models" / "sd15",
        device=device,
    ).to(device)

    image = Image.new("RGB", (224, 224), color=(180, 160, 120))
    out = model.infer_full(
        image=image,
        snr_db=5.0,
        sd_height=256,
        sd_width=256,
        sd_steps=1,
        sd_guidance=7.5,
        return_debug=True,
    )

    if not isinstance(out["recovered_text"], str) or len(out["recovered_text"].strip()) == 0:
        raise RuntimeError("Real CKB system forward recovered_text invalid.")
    if out["reconstructed_image"].size != (256, 256):
        raise RuntimeError("Real CKB system forward reconstructed image size mismatch.")
    print("[CHECK] real CKB system forward: OK")


def check_nam_snr_sensitivity() -> None:
    nam = NAM(feature_dim=128)
    nam.eval()
    x = torch.randn(2, 16, 128)
    with torch.no_grad():
        out_low = nam(x, snr=0.0)
        out_high = nam(x, snr=10.0)
    if torch.allclose(out_low, out_high):
        raise RuntimeError("NAM output does not change with SNR.")
    print("[CHECK] NAM SNR sensitivity: OK")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    check_ckb_assets(base_dir)
    check_channel_numerics()
    check_nam_snr_sensitivity()
    check_system_forward(base_dir)
    check_system_forward_real_ckb(base_dir)
    check_real_ckb_inference(base_dir)
    print("[CHECK] overall system: PASS")


if __name__ == "__main__":
    main()
