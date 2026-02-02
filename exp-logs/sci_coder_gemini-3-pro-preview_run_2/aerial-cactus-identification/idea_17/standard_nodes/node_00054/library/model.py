import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class WideBasicBlock(nn.Module):
    """
    Standard Residual Block for Wide ResNet.
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(WideBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=Config.KERNEL_SIZE,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=Config.KERNEL_SIZE,
            stride=1,
            padding=1,
            bias=False,
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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class HybridCactusClassifier(nn.Module):
    """
    Custom Narrow ResNet with Multi-Scale Aggregation.
    Optimized for low-resolution 32x32 images.
    Cite: solution_lesson_node_00016, solution_lesson_node_00019
    """

    def __init__(self):
        super(HybridCactusClassifier, self).__init__()

        # Configuration
        channels = Config.BACKBONE_CHANNELS  # [16, 32, 64]

        # Initial Convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(
                Config.INPUT_CHANNELS,
                channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 16 channels, 32x32 resolution
        self.layer1 = self._make_layer(channels[0], channels[0], stride=1, num_blocks=2)

        # Stage 2: 32 channels, 16x16 resolution (Stride 2)
        self.layer2 = self._make_layer(channels[0], channels[1], stride=2, num_blocks=2)

        # Stage 3: 64 channels, 8x8 resolution (Stride 2)
        self.layer3 = self._make_layer(channels[1], channels[2], stride=2, num_blocks=2)

        # --- Multi-Scale Head ---
        # Aggregates GAP from Stage 2 and Stage 3
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Input to FC is channels[1] + channels[2]
        self.fc = nn.Linear(channels[1] + channels[2], Config.NUM_CLASSES)

        # Initialization
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, stride, num_blocks):
        layers = []
        layers.append(WideBasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(WideBasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial Conv
        x = self.init_conv(x)

        # Stage 1
        x1 = self.layer1(x)

        # Stage 2
        x2 = self.layer2(x1)

        # Stage 3
        x3 = self.layer3(x2)

        # --- Multi-Scale Aggregation ---

        # Stage 2 Features: (B, 32, 16, 16) -> GAP -> (B, 32)
        feat2 = torch.flatten(self.pool(x2), 1)

        # Stage 3 Features: (B, 64, 8, 8) -> GAP -> (B, 64)
        feat3 = torch.flatten(self.pool(x3), 1)

        # Concatenate
        combined_feat = torch.cat([feat2, feat3], dim=1)

        # Classification
        logits = self.fc(combined_feat)

        return logits
