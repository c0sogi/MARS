import torch
import torch.nn as nn
from library.config import Z_DIM


class ResidualBlock(nn.Module):
    """
    Residual Block with Dilated Convolutions.
    """

    def __init__(self, channels, dilation):
        super(ResidualBlock, self).__init__()
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
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class InkDetectorFCN(nn.Module):
    """
    Residual FCN with Dilated Convolutions for Ink Detection.
    Cite solution_lesson_node_00003: Uses dilated convolutions to expand receptive field.
    """

    def __init__(self, in_channels=Z_DIM, compression_dim=32):
        """
        Initialize the InkDetectorFCN model.
        """
        super(InkDetectorFCN, self).__init__()

        # 1. Learnable Depth Compression (Bottleneck)
        # Cite solution_lesson_node_00002: Projects 3D volume to 2D features.
        self.depth_compress = nn.Sequential(
            nn.Conv2d(in_channels, compression_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(compression_dim),
            nn.ReLU(inplace=True),
        )

        # 2. Context Block with Dilated Residuals
        # Cite solution_lesson_node_00003: Increasing dilation rates (1, 2, 4, 8)
        self.context_block = nn.Sequential(
            ResidualBlock(compression_dim, dilation=1),
            ResidualBlock(compression_dim, dilation=2),
            ResidualBlock(compression_dim, dilation=4),
            ResidualBlock(compression_dim, dilation=8),
        )

        # 3. Output Layer
        self.output_head = nn.Sequential(
            nn.Conv2d(compression_dim, 1, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.depth_compress(x)
        x = self.context_block(x)
        x = self.output_head(x)
        return x
