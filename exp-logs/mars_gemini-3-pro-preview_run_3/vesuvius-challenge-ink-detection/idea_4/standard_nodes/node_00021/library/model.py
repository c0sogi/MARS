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


class InkDetector(nn.Module):
    """
    Lean Residual FCN with Dilated Convolutions.
    Cite Lesson 00004: Lean Spatial Backbones in 2.5D Architectures Improve Generalization.
    Cite Lesson 00015: Sequential Hierarchy vs. Parallel Multi-Scale Context.
    """

    def __init__(self):
        super(InkDetector, self).__init__()

        # 1. Learnable 2.5D Projection
        # Compresses Z_DIM (65) -> PROJECTION_DIM
        # Cite Lesson 00002: 2.5D Volumetric Projection via Learnable Bottlenecks
        self.projection = nn.Sequential(
            nn.Conv2d(Config.Z_DIM, Config.PROJECTION_DIM, kernel_size=1, bias=False),
            nn.BatchNorm2d(Config.PROJECTION_DIM),
            nn.ReLU(inplace=True),
        )

        # Adapter / Initial Conv
        self.adapter = nn.Sequential(
            nn.Conv2d(
                Config.PROJECTION_DIM,
                Config.BACKBONE_CHANNELS,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Config.BACKBONE_CHANNELS),
            nn.ReLU(inplace=True),
        )

        # 2. Sequential Hierarchical Backbone
        # Stack of DilatedResBlocks
        # Cite Lesson 00003: Expanding Receptive Fields via Dilated Convolutions
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

        # Adapter
        x = self.adapter(x)  # (B, BACKBONE_CHANNELS, H, W)

        # Backbone
        x = self.backbone(x)  # (B, BACKBONE_CHANNELS, H, W)

        # Classifier
        logits = self.classifier(x)  # (B, 1, H, W)

        return logits
