"""WITT baseline model scaffold (ViT-based) for Fig.10.

Paper traceability:
- WITT baseline comparison in Fig.10 is 论文明确写出.
- Specific architecture/hparams are 为复现做的合理实现选择.
"""

import torch
from torch import Tensor, nn


class WITTBaseline(nn.Module):
    """ViT-based WITT baseline interface."""

    def __init__(self, patch_size: int = 16, d_model: int = 128, nhead: int = 8) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
        )
        self.recon_head = nn.ConvTranspose2d(d_model, 3, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: Tensor, snr: Tensor) -> Tensor:
        """Return reconstructed images [B,3,H,W]."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images [B,3,H,W], got {tuple(images.shape)}")
        bsz = images.shape[0]
        x = self.patch_embed(images)
        h, w = x.shape[2], x.shape[3]
        tokens = x.flatten(2).transpose(1, 2)
        noise_scale = (1.0 / (snr.view(-1, 1, 1).abs() + 1.0)).to(images.dtype)
        tokens = tokens + torch.randn_like(tokens) * noise_scale
        tokens = self.transformer(tokens)
        feat = tokens.transpose(1, 2).reshape(bsz, self.d_model, h, w)
        recon = torch.sigmoid(self.recon_head(feat))
        return recon
