import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResBlock(nn.Module):
    """
    A Residual Block with dilated convolutions to maintain resolution
    while expanding the receptive field.
    """

    def __init__(self, channels, dilation):
        super(DilatedResBlock, self).__init__()
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


class HDNPCA(nn.Module):
    """
    Lean Residual Fully Convolutional Network.
    Uses a 2.5D projection followed by a stack of dilated residual blocks.
    Optimized for high throughput and generalization (Cite Lesson 00004).
    """

    def __init__(self):
        super(HDNPCA, self).__init__()

        # 1. Learnable 2.5D Projection
        # Compresses Z_DIM (65) -> PROJECTION_DIM
        self.projection = nn.Sequential(
            nn.Conv2d(Config.Z_DIM, Config.PROJECTION_DIM, kernel_size=1, bias=False),
            nn.BatchNorm2d(Config.PROJECTION_DIM),
            nn.ReLU(inplace=True),
        )

        # 2. Sequential Hierarchical Backbone
        # Stack of DilatedResBlocks (Cite Lesson 00003)
        layers = []
        for rate in Config.BACKBONE_DILATIONS:
            layers.append(DilatedResBlock(Config.BACKBONE_CHANNELS, dilation=rate))
        self.backbone = nn.Sequential(*layers)

        # 3. Classifier
        self.classifier = nn.Conv2d(Config.BACKBONE_CHANNELS, 1, kernel_size=1)

    def forward(self, x):
        # x shape: (Batch, Z_DIM, H, W)

        # Projection
        x = self.projection(x)  # (B, PROJ_DIM, H, W)

        # Backbone
        x = self.backbone(x)  # (B, BACKBONE_CHANNELS, H, W)

        # Classifier
        logits = self.classifier(x)  # (B, 1, H, W)

        return logits
