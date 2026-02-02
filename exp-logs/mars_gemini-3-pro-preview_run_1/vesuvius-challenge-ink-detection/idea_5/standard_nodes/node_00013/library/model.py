import torch
import torch.nn as nn
from library import config


class ResBlock(nn.Module):
    """
    Residual Block with support for dilation to preserve resolution
    while expanding receptive field.
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add -> ReLU
    """

    def __init__(self, channels, dilation=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class SFRPNet(nn.Module):
    """
    Split-Frequency Resolution-Preserving Network (SFRP-Net).

    Architecture:
    1. Depth-Projection Stem: Projects 3D slices to feature space using Instance Norm.
    2. Dual-Stream Backbone:
       - Stream A: High-Fidelity Texture (Dilation=1).
       - Stream B: Structural Context (Increasing Dilation).
    3. Fusion Head: Concatenates streams and projects to probability map logits.
    """

    def __init__(self):
        super(SFRPNet, self).__init__()

        # --- 1. Depth-Projection Stem ---
        # Projects Z_DIM slices to MODEL_CHANNELS
        # Uses InstanceNorm to decouple from global absolute intensity
        self.stem = nn.Sequential(
            nn.Conv2d(config.Z_DIM, config.MODEL_CHANNELS, kernel_size=1, bias=False),
            nn.InstanceNorm2d(config.MODEL_CHANNELS, affine=True),
            nn.ReLU(inplace=True),
        )

        # --- 2. Dual-Stream Backbone ---

        # Stream A: High-Fidelity Texture
        # Maintains local focus with fixed dilation=1.
        # We match the depth of Stream B (number of blocks).
        num_blocks = len(config.STREAM_B_DILATIONS)
        self.stream_a = nn.ModuleList(
            [ResBlock(config.MODEL_CHANNELS, dilation=1) for _ in range(num_blocks)]
        )

        # Stream B: Structural Context
        # Expands receptive field exponentially based on config
        self.stream_b = nn.ModuleList(
            [
                ResBlock(config.MODEL_CHANNELS, dilation=d)
                for d in config.STREAM_B_DILATIONS
            ]
        )

        # --- 3. Fusion Head ---
        # Concatenates outputs of A and B, then projects to 1 channel.
        # Input channels = MODEL_CHANNELS * 2 (from concat)
        self.fusion = nn.Conv2d(config.MODEL_CHANNELS * 2, 1, kernel_size=1)

    def forward(self, x):
        # Input shape: (Batch, Z_DIM, H, W)

        # Stem
        x = self.stem(x)

        # Parallel Processing

        # Stream A
        feat_a = x
        for block in self.stream_a:
            feat_a = block(feat_a)

        # Stream B
        feat_b = x
        for block in self.stream_b:
            feat_b = block(feat_b)

        # Fusion
        # Concatenate along channel dimension
        combined = torch.cat([feat_a, feat_b], dim=1)

        # Final projection to logits
        # Note: Sigmoid is applied during inference or via BCEWithLogitsLoss during training
        logits = self.fusion(combined)

        return logits
