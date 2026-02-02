import torch
import torch.nn as nn
from library.config import Z_DIM


class InkDetectorFCN(nn.Module):
    """
    Shallow Fully Convolutional Network (FCN) for Ink Detection.

    This model operates on full-resolution 3D surface volumes (Z-slices).
    It uses a learnable 1x1 convolution to compress the depth dimension,
    followed by a stack of 3x3 convolutions to capture spatial context,
    and a final 1x1 convolution for pixel-wise binary classification.
    """

    def __init__(self, in_channels=Z_DIM, compression_dim=16):
        """
        Initialize the InkDetectorFCN model.

        Args:
            in_channels (int): Number of input channels (Z-slices). Defaults to Z_DIM (65).
            compression_dim (int): Number of channels after the depth compression layer.
        """
        super(InkDetectorFCN, self).__init__()

        # 1. Learnable Depth Compression (Bottleneck)
        # Reduces the high-dimensional Z-axis input (65) to a smaller feature set (e.g., 16).
        # This allows the network to learn how to weight different depths.
        self.depth_compress = nn.Sequential(
            nn.Conv2d(in_channels, compression_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(compression_dim),
            nn.ReLU(inplace=True),
        )

        # 2. Spatial Context Block
        # A stack of standard 3x3 convolutions to capture the local spatial continuity
        # of ink strokes. Padding is set to 1 to maintain the spatial resolution.
        self.spatial_block = nn.Sequential(
            # Layer 1: Expand to 32 filters
            nn.Conv2d(compression_dim, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Layer 2: Expand to 64 filters for richer feature representation
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Layer 3: Compress back to 32 filters
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 3. Output Layer
        # Projects features to a single channel probability map.
        self.output_head = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Z_DIM, Height, Width).

        Returns:
            torch.Tensor: Output probability map of shape (Batch, 1, Height, Width).
        """
        # Apply depth compression
        x = self.depth_compress(x)

        # Apply spatial convolutions
        x = self.spatial_block(x)

        # Generate final probabilities
        x = self.output_head(x)

        return x
