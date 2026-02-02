import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResidualBlock(nn.Module):
    """
    A Residual Block with Dilated Convolutions and Batch Normalization.
    Maintains full spatial resolution (padding = dilation).
    """

    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Identity mapping adjustment if channel dimensions change
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.act(out)
        return out


class InkDetector(nn.Module):
    """
    Sequential Dilated FCN (Cite Lesson 00035).
    Replaces U-Net skip connections with a purely sequential hierarchical backbone
    to avoid propagating noise in low-SNR volumetric data.
    """

    def __init__(self):
        super(InkDetector, self).__init__()

        # --- 1. Learnable 2.5D Projection ---
        self.projection = nn.Sequential(
            nn.Conv2d(
                Config.Z_DEPTH, Config.PROJECTION_CHANNELS, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(Config.PROJECTION_CHANNELS),
            nn.ReLU(inplace=True),
        )

        # --- 2. Dilated Encoder (Backbone) ---
        self.encoder_blocks = nn.ModuleList()

        in_ch = Config.PROJECTION_CHANNELS
        out_ch = Config.BACKBONE_CHANNELS

        for dilation in Config.DILATION_RATES:
            block = DilatedResidualBlock(in_ch, out_ch, dilation=dilation)
            self.encoder_blocks.append(block)
            in_ch = out_ch

        # --- 3. Classification Head ---
        self.classifier = nn.Conv2d(out_ch, 1, kernel_size=1)

    def forward(self, x):
        # Handle potential extra channel dim
        if x.dim() == 5:
            x = x.squeeze(1)

        # 1. Projection
        x = self.projection(x)

        # 2. Sequential Backbone
        for block in self.encoder_blocks:
            x = block(x)

        # 3. Classification
        logits = self.classifier(x)

        return logits
