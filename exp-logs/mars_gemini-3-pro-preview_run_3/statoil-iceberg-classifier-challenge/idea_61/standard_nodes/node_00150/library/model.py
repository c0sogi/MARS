import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class EnergySE(nn.Module):
    """
    Energy-based Squeeze-and-Excitation Module.
    Replaces Global Average Pooling with Global L2 Pooling to capture
    total channel activity (positive peaks and negative shadows).
    """

    def __init__(self, channels, reduction=16):
        super(EnergySE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(
            1
        )  # Used as helper for shape, calculation is manual
        reduced_channels = max(channels // reduction, 4)

        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Global L2 Pooling: sqrt(mean(x^2))
        # We add epsilon for numerical stability
        y = x.pow(2).mean(dim=(2, 3)).sqrt().view(b, c)

        # Excitation
        y = self.fc(y).view(b, c, 1, 1)

        # Scale
        return x * y


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block:
    Conv2d (Bias=True) -> BN -> LeakyReLU -> EnergySE -> MaxPool
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.se = EnergySE(out_channels)
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class EA_IDPH_CNN(nn.Module):
    """
    Energy-Attentive Isomorphic CNN.

    Architecture:
    1. 4-Stage Plain CNN Backbone with Energy-SE.
    2. Corrected Isomorphic Readout from Stage 3 and Stage 4.
    3. Fusion with raw incidence angle.
    """

    def __init__(self):
        super(EA_IDPH_CNN, self).__init__()

        # --- Backbone ---
        # Input: 3 channels (HH, HV, Avg) -> 75x75
        self.block1 = ConvBlock(Config.IN_CHANNELS, 64)  # -> 37x37
        self.block2 = ConvBlock(64, 128)  # -> 18x18
        self.block3 = ConvBlock(128, 128)  # -> 9x9 (Stage 3)
        self.block4 = ConvBlock(128, 128)  # -> 4x4 (Stage 4)

        # --- Isomorphic Readout Projections ---
        # Decoupled projections for Stage 3 and Stage 4
        # Reduces 128 channels to 64 before pooling
        self.proj3 = nn.Conv2d(128, 64, kernel_size=1)
        self.proj4 = nn.Conv2d(128, 64, kernel_size=1)

        # --- Classification Head ---
        # Features:
        # Stage 3: 64 (Max) + 64 (Min) = 128
        # Stage 4: 64 (Max) + 64 (Min) = 128
        # Angle: 1
        # Total: 257
        self.head = nn.Sequential(
            nn.Linear(256 + 1, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, Config.NUM_CLASSES),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        # --- Backbone Forward ---
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)  # Stage 3 Output
        x4 = self.block4(x3)  # Stage 4 Output

        # --- Isomorphic Readout ---

        # Stage 3 Processing
        p3 = self.proj3(x3)
        # Global Max Pooling
        max3 = F.adaptive_max_pool2d(p3, 1).view(p3.size(0), -1)
        # Global Min Pooling (implemented as -Max(-x))
        min3 = -F.adaptive_max_pool2d(-p3, 1).view(p3.size(0), -1)

        # Stage 4 Processing
        p4 = self.proj4(x4)
        max4 = F.adaptive_max_pool2d(p4, 1).view(p4.size(0), -1)
        min4 = -F.adaptive_max_pool2d(-p4, 1).view(p4.size(0), -1)

        # Concatenate Features
        features = torch.cat([max3, min3, max4, min4], dim=1)  # Size: Batch x 256

        # --- Fusion with Angle ---
        # Ensure angle is (Batch, 1)
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        combined = torch.cat([features, angle], dim=1)  # Size: Batch x 257

        # --- Classification ---
        logits = self.head(combined)

        return logits.squeeze(1)
