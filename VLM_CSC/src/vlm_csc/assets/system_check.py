"""System check utilities for VLM-CSC."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def check_channel_numerics() -> None:
    """Verify channel model numerics."""
    from vlm_csc.models.channel import PhysicalChannel

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


def check_ckb_assets(base_dir: Path | str) -> None:
    """Verify CKB assets exist."""
    base_dir = Path(base_dir)
    models_dir = base_dir / "data" / "assets" / "downloaded_models"
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


def check_nam_snr_sensitivity() -> None:
    """Verify NAM responds to SNR changes."""
    from vlm_csc.models.nam import NAM

    nam = NAM(feature_dim=128)
    nam.eval()
    x = torch.randn(2, 16, 128)
    with torch.no_grad():
        out_low = nam(x, snr=0.0)
        out_high = nam(x, snr=10.0)
    if torch.allclose(out_low, out_high):
        raise RuntimeError("NAM output does not change with SNR.")
    print("[CHECK] NAM SNR sensitivity: OK")


def check_system_forward(blip_dir: Path | str, sd_dir: Path | str) -> None:
    """Verify system forward pass works."""
    from vlm_csc import VLMCscSystem

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VLMCscSystem(
        feature_dim=128,
        max_text_len=24,
        channel_type="awgn",
        use_real_ckb=False,
        enable_med=False,
        blip_dir=blip_dir,
        sd_dir=sd_dir,
        device=device,
    ).to(device)

    image = Image.new("RGB", (224, 224), color=(110, 130, 170))

    out = model.forward_text_train(
        image=image,
        snr_db=5.0,
        image_id="img_0",
        dataset_id="smoke",
    )
    logits = out["logits"]
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("Train forward logits type invalid.")
    if logits.shape[:2] != (1, 24):
        raise RuntimeError(f"Train forward logits shape invalid: {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Train forward logits contain non-finite values.")

    infer_out = model.infer_full(image=image, snr_db=5.0)
    recovered_text = infer_out["recovered_text"]

    if not isinstance(recovered_text, str):
        raise RuntimeError("Recovered text type invalid.")
    if len(recovered_text) == 0:
        raise RuntimeError("Recovered text is empty.")

    print("[CHECK] system forward: OK")


def main() -> None:
    """Run all system checks."""
    check_channel_numerics()
    check_nam_snr_sensitivity()
    print("[CHECK] basic checks: PASS")


if __name__ == "__main__":
    main()
