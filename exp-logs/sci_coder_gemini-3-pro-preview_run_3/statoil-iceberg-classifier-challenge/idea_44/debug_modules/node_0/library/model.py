import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class HybridSE(nn.Module):
    """
    Hybrid Squeeze-and-Excitation Module.
    Uses Global Average Pooling to summarize channel statistics.
    """

    def __init__(self, channels, reduction=16):
        super(HybridSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for Plain CNN.
    Structure: Conv2d -> BN -> LeakyReLU -> SE -> MaxPool
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        # Explicitly retain bias as per solution design
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=config.LEAKY_RELU_SLOPE, inplace=True)
        self.se = HybridSE(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class SPPCNN(nn.Module):
    """
    Split-Polarity Plain CNN (SPP-CNN).
    Features a 4-stage backbone with a split readout for Peak and Shadow features.
    """

    def __init__(self):
        super(SPPCNN, self).__init__()

        # --- Backbone (4 Stages) ---
        # Input: 3 channels (HH, HV, Avg) -> 75x75

        # Stage 1: 3 -> 64
        self.block1 = ConvBlock(config.NUM_INPUT_CHANNELS, 64)

        # Stage 2: 64 -> 128
        self.block2 = ConvBlock(64, 128)

        # Stage 3: 128 -> 128
        self.block3 = ConvBlock(128, 128)

        # Stage 4: 128 -> 128
        self.block4 = ConvBlock(128, 128)

        # --- Split-Branch Readout ---
        # Branch 1: Peak Features (High Backscatter)
        self.peak_conv = nn.Conv2d(128, 64, kernel_size=1, bias=True)

        # Branch 2: Shadow Features (Signal Voids)
        self.shadow_conv = nn.Conv2d(128, 64, kernel_size=1, bias=True)

        # --- Classification Head ---
        # Inputs: 64 (Peak) + 64 (Shadow) + 1 (Angle) = 129
        self.head = nn.Sequential(
            nn.Linear(129, 256),
            nn.LeakyReLU(negative_slope=config.LEAKY_RELU_SLOPE, inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(256, 1),
        )

    def forward(self, x, angle):
        # Backbone Forward Pass
        x = self.block1(x)  # 75 -> 37
        x = self.block2(x)  # 37 -> 18
        x = self.block3(x)  # 18 -> 9
        x = self.block4(x)  # 9 -> 4

        # Split Readout

        # 1. Peak Branch: Learn features for peaks, then Global Max Pool
        x_peak = self.peak_conv(x)
        x_peak = F.adaptive_max_pool2d(x_peak, 1).view(x_peak.size(0), -1)

        # 2. Shadow Branch: Learn features for shadows, then Global Min Pool
        # Implemented as Max(-x) to extract magnitude of most negative values
        x_shadow = self.shadow_conv(x)
        x_shadow = F.adaptive_max_pool2d(-x_shadow, 1).view(x_shadow.size(0), -1)

        # Feature Fusion
        # Ensure angle is (Batch, 1)
        angle = angle.view(-1, 1)

        # Concatenate: [Peak Features, Shadow Features, Incidence Angle]
        features = torch.cat([x_peak, x_shadow, angle], dim=1)

        # Classification
        out = self.head(features)

        return out
