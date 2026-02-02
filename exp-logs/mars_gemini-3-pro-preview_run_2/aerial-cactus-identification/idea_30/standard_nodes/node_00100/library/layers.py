import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA) module.

    References:
        Wang et al., "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks", CVPR 2020.
    """

    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        # Adaptive kernel size calculation
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg_pool(x)  # (B, C, 1, 1)

        # Reshape for 1D Conv: (B, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)

        y = self.conv(y)  # (B, 1, C)

        # Reshape back: (B, C, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)
        return x * y.expand_as(x)
