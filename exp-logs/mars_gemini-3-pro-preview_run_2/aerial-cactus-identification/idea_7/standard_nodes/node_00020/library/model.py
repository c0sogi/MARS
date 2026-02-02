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


class MultiScaleResNet(nn.Module):
    """
    Multi-Scale ResNet Architecture for Cactus Identification.
    Aggregates features from intermediate and final stages to capture multi-scale information
    efficiently without a decoder.
    """

    def __init__(self):
        super(MultiScaleResNet, self).__init__()
        c = Config.ENCODER_CHANNELS

        # --- Encoder (Backbone) ---
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, c[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32x32 -> 32x32
        self.layer1 = ResidualBlock(c[0], c[0], stride=1)

        # Stage 2: 32x32 -> 16x16
        self.layer2 = ResidualBlock(c[0], c[1], stride=2)

        # Stage 3: 16x16 -> 8x8
        self.layer3 = ResidualBlock(c[1], c[2], stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Input to FC is concatenation of stage 2 and stage 3 features
        fc_in_features = c[1] + c[2]
        self.fc = nn.Linear(fc_in_features, Config.NUM_CLASSES)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layer1(x)

        e2 = self.layer2(x)  # [B, 32, 16, 16]
        e3 = self.layer3(e2)  # [B, 64, 8, 8]

        # Multi-scale aggregation
        # GAP on e2
        out2 = self.global_pool(e2).view(e2.size(0), -1)
        # GAP on e3
        out3 = self.global_pool(e3).view(e3.size(0), -1)

        # Concatenate
        out = torch.cat([out2, out3], dim=1)

        logits = self.fc(out)
        return logits
