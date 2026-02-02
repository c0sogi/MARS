import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResidualBlock(nn.Module):
    """
    A Residual Block with Dilated Convolutions and Group Normalization.
    Maintains full spatial resolution (padding = dilation).
    """

    def __init__(self, in_channels, out_channels, dilation, groups=8):
        super(DilatedResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.act = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(groups, out_channels)

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
        out = self.gn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += residual
        out = self.act(out)
        return out


class DilatedFCN(nn.Module):
    """
    Sequential Dilated Fully Convolutional Network.
    Cite {solution_lesson_node_00035}: Removes U-Net skip connections to prevent noise propagation.
    Cite {solution_lesson_node_00036}: Uses deep dilated hierarchy (r=16) and wide backbone (64ch).

    Architecture:
    1. Projection: Compress Z-depth (65) -> Projection Channels.
    2. Backbone: Sequential Dilated Residual Blocks (r=1, 2, 4, 8, 16).
    3. Head: 1x1 Conv to binary logits.
    """

    def __init__(self):
        super(DilatedFCN, self).__init__()

        # --- 1. Learnable 2.5D Projection ---
        self.projection = nn.Sequential(
            nn.Conv2d(
                Config.Z_DEPTH, Config.PROJECTION_CHANNELS, kernel_size=1, bias=False
            ),
            nn.GroupNorm(Config.GROUP_NORM_GROUPS, Config.PROJECTION_CHANNELS),
            nn.ReLU(inplace=True),
        )

        # --- 2. Dilated Backbone ---
        self.backbone = nn.Sequential()

        in_ch = Config.PROJECTION_CHANNELS
        out_ch = Config.BACKBONE_CHANNELS

        for i, dilation in enumerate(Config.DILATION_RATES):
            block = DilatedResidualBlock(
                in_ch, out_ch, dilation=dilation, groups=Config.GROUP_NORM_GROUPS
            )
            self.backbone.add_module(f"block_{i}_d{dilation}", block)
            in_ch = out_ch

        # --- 3. Classification Head ---
        self.classifier = nn.Conv2d(Config.BACKBONE_CHANNELS, 1, kernel_size=1)

    def forward(self, x):
        # Handle potential extra channel dim from dataloaders
        if x.dim() == 5:
            x = x.squeeze(1)

        x = self.projection(x)
        x = self.backbone(x)
        logits = self.classifier(x)

        return logits
