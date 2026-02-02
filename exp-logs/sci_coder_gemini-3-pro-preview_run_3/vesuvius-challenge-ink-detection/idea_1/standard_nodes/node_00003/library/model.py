import torch
import torch.nn as nn
from library.config import Z_DIM


class ResidualBlock(nn.Module):
    """
    Simple Residual Block with dilated convolutions for context aggregation.
    """

    def __init__(self, channels, dilation=1):
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
        return self.relu(out)


class InkDetectorFCN(nn.Module):
    """
    Residual Fully Convolutional Network (FCN) for Ink Detection.

    Uses a learnable 1x1 depth compression followed by a stack of
    dilated residual blocks to capture multi-scale spatial context.
    """

    def __init__(self, in_channels=Z_DIM, compression_dim=32):
        """
        Initialize the InkDetectorFCN model.

        Args:
            in_channels (int): Number of input channels (Z-slices). Defaults to Z_DIM (65).
            compression_dim (int): Number of channels after the depth compression layer.
        """
        super(InkDetectorFCN, self).__init__()

        # 1. Learnable Depth Compression (Bottleneck)
        # Increased compression dim to 32 to preserve more Z-axis information.
        self.depth_compress = nn.Sequential(
            nn.Conv2d(in_channels, compression_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(compression_dim),
            nn.ReLU(inplace=True),
        )

        # 2. Spatial Context Block (Residual + Dilated)
        # Uses dilated convolutions to increase receptive field without downsampling.
        self.spatial_block = nn.Sequential(
            # Initial expansion
            nn.Conv2d(compression_dim, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Residual Blocks with increasing dilation
            ResidualBlock(64, dilation=1),
            ResidualBlock(64, dilation=2),
            ResidualBlock(64, dilation=4),
            ResidualBlock(64, dilation=8),  # Large context
            # Compression back to 32
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 3. Output Layer
        self.output_head = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        """
        Forward pass of the network.
        """
        x = self.depth_compress(x)
        x = self.spatial_block(x)
        x = self.output_head(x)
        return x
