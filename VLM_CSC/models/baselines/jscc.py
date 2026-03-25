"""JSCC baseline model scaffold (CNN-based) for Fig.10.

Paper traceability:
- JSCC baseline comparison in Fig.10 is 论文明确写出.
- Specific architecture/hparams are 为复现做的合理实现选择.
"""

from torch import Tensor, nn
import torch


class JSCCBaseline(nn.Module):
    """CNN-based JSCC baseline interface."""

    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, images: Tensor, snr: Tensor) -> Tensor:
        """Return reconstructed images [B,3,H,W]."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images [B,3,H,W], got {tuple(images.shape)}")
        latent = self.encoder(images)
        noise_scale = (1.0 / (snr.view(-1, 1, 1, 1).abs() + 1.0)).to(images.dtype)
        latent = latent + torch.randn_like(latent) * noise_scale
        recon = self.decoder(latent)
        return recon
