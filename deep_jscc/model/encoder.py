import torch
import torch.nn as nn


class Encoder(nn.Module):
    """DeepJSCC 编码器（严格遵守论文架构图）。

    论文架构（Fig.2）:
      1. Normalization: 像素归一化 [0,255] → [0,1]
      2. 五层卷积（全部PReLU激活）
      3. Normalization: 功率归一化 E[|z|²] = 1

    关键约束：
      - 全卷积结构（Fully Convolutional），不使用全连接层。
      - 默认路径不使用池化；空间降维仅通过 stride>1 的卷积实现。
      - 输出为连续实值信道符号（real-valued），不做复数/离散调制。
      - 压缩率仅由 `c` 控制：输出形状恒为 (B, c, H_out, W_out)。
      - 功率归一化只约束平均发射功率，不改变维度、不引入离散化。

    参数：
        - in_channels (int): 输入图像通道数（例如 RGB=3）。
        - c (int): 每个空间位置的信道符号维度（唯一的显式码率/带宽控制参数）。
        - p (float): 每个实值信道符号的平均功率，用于功率归一化。
        - input_normalized (bool): 输入是否已归一化到[0,1]。
            True: 输入已通过ToTensor归一化（默认）
            False: 输入在[0,255]范围，Encoder内部会归一化
    """

    def __init__(self, in_channels: int = 3, c: int = 128, p: float = 1.0, input_normalized: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.c = int(c)
        self.p = float(p)
        self.input_normalized = input_normalized  # 输入是否已归一化到[0,1]

        # 五个卷积层（严格按照论文图结构）
        # 5x5x16/2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.PReLU(),
        )
        # 5x5x32/2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.PReLU(),
        )
        # 5x5x32/1
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
        )
        # 5x5x32/1
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
        )
        # 5x5xc/1
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, self.c, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 显式约束输入格式为 (B, C, H, W)
        if x.dim() != 4:
            raise ValueError("Encoder expects a 4D tensor in (B, C, H, W) format.")
        if x.shape[1] != self.in_channels:
            raise ValueError("Encoder input channel mismatch: expected C=%d, got C=%d." % (self.in_channels, x.shape[1]))

        # 输入归一化（论文架构图第一层：Normalization）
        # 论文："input images are normalized by the maximum pixel value 255, producing pixel values in the [0, 1] range"
        if not self.input_normalized:
            # 如果输入在[0, 255]范围，归一化到[0, 1]
            x = x / 255.0
        # 如果input_normalized=True，说明输入已通过ToTensor归一化，无需再处理

        # 前向传播（5层卷积，已包含PReLU）
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # 功率归一化（论文图中最后的Normalization层）
        # 输出为连续实值信道符号：形状 (B, c, H_out, W_out)
        H_out = x.shape[2]
        W_out = x.shape[3]
        K = self.c * H_out * W_out

        # 功率归一化：约束平均发射功率，不改变维度
        # 使用更大的eps防止数值不稳定（从1e-8提升到1e-3）
        eps = 1e-4
        energy = x.pow(2).sum(dim=(1, 2, 3), keepdim=True)
        # 使用clamp限制scale的范围，防止极端值
        scale = (K * self.p) ** 0.5 / (energy + eps).sqrt()
        scale = torch.clamp(scale, max=100.0)  # 限制最大缩放倍数
        x = x * scale

        return x


