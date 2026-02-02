import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class WideSE(nn.Module):
    """
    Wide-Attention Squeeze-and-Excitation Module.
    Uses a low reduction ratio (r=2) to prevent information bottlenecks.
    """

    def __init__(self, in_channels, reduction=Config.SE_REDUCTION_RATIO):
        super(WideSE, self).__init__()
        mid_channels = in_channels // reduction

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.LeakyReLU(negative_slope=Config.LEAKY_RELU_SLOPE, inplace=True),
            nn.Linear(mid_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for the Plain CNN Backbone.
    Structure: Conv2d(Bias=True) -> BN -> LeakyReLU -> WideSE -> MaxPool.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        # Explicitly retain bias=True to preserve initialization dynamics
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=Config.LEAKY_RELU_SLOPE, inplace=True)
        self.se = WideSE(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class WA_IDPH_CNN(nn.Module):
    """
    Wide-Attention Isomorphic Dual-Polarity CNN.
    Features:
    1. 4-Stage Plain CNN Backbone.
    2. Decoupled Isomorphic Readout (Stage 3 & 4).
    3. Raw Angle Fusion.
    """

    def __init__(self):
        super(WA_IDPH_CNN, self).__init__()

        # --- Backbone ---
        # Input: (3, 75, 75)
        # Stage 1: 3 -> 64
        self.stage1 = ConvBlock(Config.INPUT_SHAPE[0], Config.BACKBONE_CHANNELS[0])
        # Stage 2: 64 -> 128
        self.stage2 = ConvBlock(
            Config.BACKBONE_CHANNELS[0], Config.BACKBONE_CHANNELS[1]
        )
        # Stage 3: 128 -> 128
        self.stage3 = ConvBlock(
            Config.BACKBONE_CHANNELS[1], Config.BACKBONE_CHANNELS[2]
        )
        # Stage 4: 128 -> 128
        self.stage4 = ConvBlock(
            Config.BACKBONE_CHANNELS[2], Config.BACKBONE_CHANNELS[3]
        )

        # --- Decoupled Isomorphic Readout ---
        # Separate 1x1 convolutions for Stage 3 and Stage 4 to learn depth-specific transformations
        self.proj3 = nn.Conv2d(Config.BACKBONE_CHANNELS[2], 64, kernel_size=1)
        self.proj4 = nn.Conv2d(Config.BACKBONE_CHANNELS[3], 64, kernel_size=1)

        # --- Classification Head ---
        # Feature Vector Calculation:
        # Stage 3: Max(64) + Min(64) = 128
        # Stage 4: Max(64) + Min(64) = 128
        # Total Image Features: 256
        # + Incidence Angle: 1
        # Total Input: 257

        self.head = nn.Sequential(
            nn.Linear(256 + 1, 256),
            nn.LeakyReLU(negative_slope=Config.LEAKY_RELU_SLOPE, inplace=True),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(256, 1),
        )

        # Note: Weights are initialized using PyTorch defaults (Kaiming Uniform)

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image tensor (B, 3, 75, 75)
            angle (torch.Tensor): Incidence angle tensor (B,)
        """
        # Backbone forward pass
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)  # Keep for readout
        x4 = self.stage4(x3)  # Keep for readout

        # --- Isomorphic Readout Stage 3 ---
        p3 = self.proj3(x3)
        # Global Max Pooling
        max3 = F.adaptive_max_pool2d(p3, 1).view(p3.size(0), -1)
        # Global Min Pooling (implemented as -max(-x))
        min3 = -F.adaptive_max_pool2d(-p3, 1).view(p3.size(0), -1)

        # --- Isomorphic Readout Stage 4 ---
        p4 = self.proj4(x4)
        # Global Max Pooling
        max4 = F.adaptive_max_pool2d(p4, 1).view(p4.size(0), -1)
        # Global Min Pooling
        min4 = -F.adaptive_max_pool2d(-p4, 1).view(p4.size(0), -1)

        # Concatenate all image features
        img_features = torch.cat([max3, min3, max4, min4], dim=1)  # (B, 256)

        # --- Feature Fusion ---
        # Reshape angle to (B, 1)
        angle = angle.view(-1, 1)

        # Concatenate raw angle with image features
        combined = torch.cat([img_features, angle], dim=1)  # (B, 257)

        # Classification
        logits = self.head(combined)

        # Return logits (shape B)
        return logits.squeeze(1)
