import torch
import torch.nn as nn
from library.config import ConvBlock


class SpatialStem(nn.Module):
    """
    Spatial Input Stem: Processes concatenated inputs with a Standard Convolution,
    Layer Normalization, and SiLU activation.

    This module handles the necessary transposition to apply LayerNorm over the
    channel dimension for (B, C, L) tensors.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        # Standard Convolution with 'same' padding for odd kernel sizes
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.ln = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # Input x: (Batch, Channels, Length)
        x = self.conv(x)

        # Transpose to (Batch, Length, Channels) for LayerNorm
        x = x.transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        # Transpose back to (Batch, Channels, Length)
        x = x.transpose(1, 2)
        return x


class DilatedResidualBlock(ConvBlock):
    """
    Dilated Residual Block with Pre-Activation structure.

    Inherits from library.config.ConvBlock which implements:
    LayerNorm -> SiLU -> Dilated Conv (k=3) -> LayerNorm -> SiLU -> Pointwise Conv (k=1) -> Dropout

    This wrapper enforces the fixed-width constraint (in_channels == out_channels)
    typical of residual backbones.
    """

    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        # Initialize the parent ConvBlock with in_channels = out_channels = channels
        super().__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            dropout=dropout,
        )


class FeedbackTCN(nn.Module):
    """
    Lightweight Residual TCN for processing recycled predictions.

    Consists of an input projection followed by a stack of DilatedResidualBlocks.
    """

    def __init__(self, in_channels=5, out_channels=32, num_blocks=3, dropout=0.1):
        super().__init__()
        # Initial projection from target dim (5) to latent dim (32)
        self.stem = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        # Stack of residual blocks with exponentially increasing dilation
        # to capture local context in the feedback signal
        blocks = []
        for i in range(num_blocks):
            dilation = 2**i
            blocks.append(
                DilatedResidualBlock(
                    channels=out_channels,
                    kernel_size=3,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        # Input x: (Batch, 5, Length)
        x = self.stem(x)
        x = self.blocks(x)
        # Output x: (Batch, 32, Length)
        return x
