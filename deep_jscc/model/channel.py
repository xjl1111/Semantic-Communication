import torch
import torch.nn as nn


class Channel(nn.Module):
    """DeepJSCC 信道层（严格三步走实现）。

    说明：
      - Encoder 输出为连续实值信道符号（real-valued tensor）。
      - Channel 仅进行物理信道扰动（噪声/衰落），不做离散调制或判决。
      - DeepJSCC 中“调制”是隐式学习行为，Channel 不引入星座映射。

    参数：
      - channel_type (str): "awgn" 或 "rayleigh"。
      - snr_db (float): 信噪比（dB），用于噪声方差计算。
      - fading (str): 仅对 Rayleigh 生效，"fast" 或 "block"。
    """

    def __init__(self, channel_type: str = "awgn", snr_db: float = 10.0, fading: str = "fast"):
        super().__init__()
        self.channel_type = str(channel_type).lower()
        self.snr_db = float(snr_db)
        self.fading = str(fading).lower()

    def forward(self, x: torch.Tensor, snr_db: float = None, seed: int = None) -> torch.Tensor:
        """对 Encoder 输出进行物理信道扰动。

        输入：
          - x: 实值张量，形状 (B, C, H, W)。
        输出：
          - 实值张量，形状与输入一致。
        """
        if x.dim() != 4:
            raise ValueError("Channel expects a 4D tensor in (B, C, H, W) format.")

        # 直接从shape提取整数
        b, c, h, w = [int(s) for s in x.shape]

        # STEP 1: 按通道维度进行 I/Q 配对（C 必须为偶数）
        if c % 2 != 0:
            raise ValueError("Channel pairing requires C to be even (I/Q pairing by channels).")

        # x -> (B, C/2, 2, H, W)：0 为实部，1 为虚部
        x_pair = x.contiguous().view(int(b), int(c // 2), 2, int(h), int(w))
        real = x_pair[:, :, 0, :, :]
        imag = x_pair[:, :, 1, :, :]
        s = torch.complex(real, imag)

        # STEP 2: 物理信道建模（AWGN / Rayleigh）
        snr = self.snr_db if snr_db is None else float(snr_db)
        # 输出已做平均功率归一化，噪声方差：sigma^2 = 10^(-SNR/10) / 2
        sigma2 = 10 ** (-snr / 10.0) / 2.0

        gen = None
        if seed is not None:
            gen = torch.Generator(device=x.device)
            gen.manual_seed(int(seed))

        noise_real = torch.randn(real.shape, device=x.device, dtype=x.dtype, generator=gen) * (sigma2 ** 0.5)
        noise_imag = torch.randn(imag.shape, device=x.device, dtype=x.dtype, generator=gen) * (sigma2 ** 0.5)
        noise = torch.complex(noise_real, noise_imag)

        if self.channel_type == "awgn":
            y = s + noise
        elif self.channel_type == "rayleigh":
            # Rayleigh 衰落：h_fading_real, h_fading_imag ~ N(0, 0.5)，E[|h_fading|^2]=1
            if self.fading not in ("fast", "block"):
                raise ValueError("fading must be 'fast' or 'block' for Rayleigh.")

            if self.fading == "fast":
                h_fading_real = torch.randn(real.shape, device=x.device, dtype=x.dtype, generator=gen) * (0.5 ** 0.5)
                h_fading_imag = torch.randn(imag.shape, device=x.device, dtype=x.dtype, generator=gen) * (0.5 ** 0.5)
            else:
                h_fading_real = torch.randn((b, 1, 1, 1), device=x.device, dtype=x.dtype, generator=gen) * (0.5 ** 0.5)
                h_fading_imag = torch.randn((b, 1, 1, 1), device=x.device, dtype=x.dtype, generator=gen) * (0.5 ** 0.5)
                # expand需要完整的目标shape
                h_fading_real = h_fading_real.expand(int(b), int(c // 2), int(h), int(w))
                h_fading_imag = h_fading_imag.expand(int(b), int(c // 2), int(h), int(w))

            h_fading = torch.complex(h_fading_real, h_fading_imag)
            y = h_fading * s + noise
        else:
            raise ValueError("channel_type must be 'awgn' or 'rayleigh'.")

        # STEP 3: 复数符号拆分并还原为实值张量 (B, C, H, W)
        y_real = y.real
        y_imag = y.imag
        y_pair = torch.stack([y_real, y_imag], dim=2)  # (B, C/2, 2, H, W)
        y_out = y_pair.contiguous().view(b, c, h, w)

        return y_out
