import torch
import torch.nn as nn


class Decoder(nn.Module):
    """DeepJSCC 解码器（论文语义一致性）。

    说明：
      - 输入为 Channel 输出的连续实值信道符号张量（不做显式解调/信道估计）。
      - 全卷积结构，不使用全连接层与池化。
      - 通过转置卷积恢复空间尺寸，与 Encoder 的 stride 下采样对齐。
      - 最终输出图像张量，默认归一化到 [0,1]（Sigmoid）。

    参数：
      - in_channels (int): 输入符号通道数 c（对应 Encoder 输出的 c）。
      - out_channels (int): 输出图像通道数（默认 3）。
      - activation (str): 输出激活函数 "sigmoid" 或 "tanh"。
    """

    def __init__(self, in_channels: int = 128, out_channels: int = 3, activation: str = "sigmoid"):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.activation = str(activation).lower()

        # 镜像Encoder结构：5层转置卷积（严格按论文图）
        # 论文图显示：4层紫色(trans conv+PReLU) + 1层绿色(trans conv+sigmoid)
        # 需要两次stride=2上采样以恢复32×32
        
        # 第1层: 5x5x32/1 trans conv + PReLU
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
        )
        # 第2层: 5x5x32/1 trans conv + PReLU
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
        )
        # 第3层: 5x5x32/1 trans conv + PReLU (还未上采样)
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
        )
        # 第4层: 5x5x16/2 trans conv + PReLU (第一次上采样: 8×8 → 16×16)
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.PReLU(),
        )
        # 第5层: 5x5x3/2 trans conv + sigmoid (第二次上采样: 16×16 → 32×32)
        self.conv5 = nn.ConvTranspose2d(16, self.out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)

        if self.activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif self.activation == "tanh":
            self.out_act = nn.Tanh()
        else:
            raise ValueError("activation must be 'sigmoid' or 'tanh'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向解码（端到端学习，不执行传统解调/译码）。

        输入：
          - x: 实值张量，形状为 (B, C, H, W)。
        输出：
          - 图像张量，形状为 (B, 3, H_img, W_img)。
        """
        # 显式约束输入格式为 (B, C, H, W)
        if x.dim() != 4:
            raise ValueError("Decoder expects a 4D tensor in (B, C, H, W) format.")
        if x.shape[1] != self.in_channels:
            raise ValueError("Decoder input channel mismatch: expected C=%d, got C=%d." % (self.in_channels, x.shape[1]))

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.out_act(x)

        return x
