import torch
import torch.nn as nn
from library.config import Config


class ResidualBlock(nn.Module):
    """
    Standard ResNet block with support for dilation.
    """

    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ResidualFCN(nn.Module):
    """
    Residual Fully Convolutional Network (FCN) with Dilated Convolutions.
    Cite {solution_lesson_node_00008}: Preserves spatial resolution (no pooling).
    Cite {solution_lesson_node_00004}: Lean backbone (32 channels) for generalization.
    """

    def __init__(self):
        super().__init__()

        # --- Hyperparameters ---
        z_dim = Config.Z_DIM
        channels = Config.MODEL_CHANNELS  # 32

        # --- 1. Depth Compression ---
        # Cite {solution_lesson_node_00002}: Learnable bottleneck
        self.depth_compression = nn.Sequential(
            nn.Conv2d(z_dim, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # --- 2. Spatial Backbone ---
        # Cite {solution_lesson_node_00003}: Dilated convolutions for texture context
        self.backbone = nn.Sequential(
            ResidualBlock(channels, dilation=1),
            ResidualBlock(channels, dilation=2),
            ResidualBlock(channels, dilation=4),
            ResidualBlock(channels, dilation=8),
        )

        # --- 3. Head ---
        self.head = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        x = self.depth_compression(x)
        x = self.backbone(x)
        return self.head(x)
