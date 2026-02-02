import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock(nn.Module):
    """
    Standard Residual Block for the ResNet Encoder.
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+) -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CactusResNet(nn.Module):
    """
    Custom ResNet Architecture for Cactus Identification.
    Uses Multi-Scale Feature Aggregation (Cite solution_lesson_node_00016)
    to combine mid-level and high-level features, avoiding the need for a decoder.
    """

    def __init__(self):
        super(CactusResNet, self).__init__()

        # Channel configuration from Config: [16, 32, 64]
        c = Config.ENCODER_CHANNELS

        # --- Encoder (Backbone) ---
        # Initial convolution to scale up from RGB
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, c[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32x32 -> 32x32 (16 channels)
        self.layer1 = ResidualBlock(c[0], c[0], stride=1)

        # Stage 2: 32x32 -> 16x16 (32 channels)
        self.layer2 = ResidualBlock(c[0], c[1], stride=2)

        # Stage 3: 16x16 -> 8x8 (64 channels)
        self.layer3 = ResidualBlock(c[1], c[2], stride=2)

        # --- Classification Head ---
        # Multi-Scale Aggregation: Concatenate GAP from Stage 2 and Stage 3
        self.fc = nn.Linear(c[1] + c[2], Config.NUM_CLASSES)

    def forward(self, x):
        # --- Encoder ---
        x0 = self.initial_conv(x)
        e1 = self.layer1(x0)
        e2 = self.layer2(e1)  # [B, 32, 16, 16]
        e3 = self.layer3(e2)  # [B, 64, 8, 8]

        # --- Head ---
        # Global Average Pooling on multi-scale features
        gap2 = F.adaptive_avg_pool2d(e2, (1, 1)).view(x.size(0), -1)  # [B, 32]
        gap3 = F.adaptive_avg_pool2d(e3, (1, 1)).view(x.size(0), -1)  # [B, 64]

        # Concatenate
        combined = torch.cat([gap2, gap3], dim=1)  # [B, 96]

        logits = self.fc(combined)
        return logits
