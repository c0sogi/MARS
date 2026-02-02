import torch
import torch.nn as nn
from library.config import Config


class DilatedResidualBlock(nn.Module):
    """
    A Residual Block with Dilated Convolutions to expand receptive field
    without losing spatial resolution.
    Structure: Conv(d) -> BN -> ReLU -> Conv(d) -> BN -> Add -> ReLU
    """

    def __init__(self, channels, dilation):
        super(DilatedResidualBlock, self).__init__()
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


class InkDetector(nn.Module):
    """
    Sequential Dilated Residual Network (InkDetector).
    Simplified architecture focusing on hierarchical feature refinement.
    Cite Lesson 00004: Lean Spatial Backbones.
    Cite Lesson 00015: Sequential Hierarchy vs. Parallel Multi-Scale.

    Features:
    1. Learnable 2.5D Projection (Depth -> Channels).
    2. Sequential Dilated Backbone (Residual connections).
    3. Simple Head (No dense aggregation).
    """

    def __init__(self):
        super(InkDetector, self).__init__()

        # 1. Learnable 2.5D Projection
        self.projection = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=Config.PROJECTION_DIM,
            kernel_size=1,
            bias=False,
        )
        self.proj_bn = nn.BatchNorm2d(Config.PROJECTION_DIM)
        self.proj_relu = nn.ReLU(inplace=True)

        # 2. Sequential Dilated Backbone
        self.backbone = nn.ModuleList()

        if Config.PROJECTION_DIM != Config.BACKBONE_WIDTH:
            raise ValueError(
                "Projection dim must match backbone width for this implementation."
            )

        for dilation in Config.DILATION_RATES:
            block = DilatedResidualBlock(Config.BACKBONE_WIDTH, dilation)
            self.backbone.append(block)

        # 3. Output Head
        # Takes the output of the last block directly
        self.head = nn.Conv2d(
            in_channels=Config.BACKBONE_WIDTH, out_channels=1, kernel_size=1
        )

    def forward(self, x):
        # Projection
        x = self.projection(x)
        x = self.proj_bn(x)
        x = self.proj_relu(x)

        # Sequential Backbone
        for block in self.backbone:
            x = block(x)

        # Output
        logits = self.head(x)

        return logits
