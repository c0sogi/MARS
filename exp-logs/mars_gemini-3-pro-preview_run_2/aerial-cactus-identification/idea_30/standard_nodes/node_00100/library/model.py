import torch
import torch.nn as nn
import torch.nn.functional as F
from library.layers import ECA
from library.config import CHANNELS, NUM_CLASSES


class BasicBlock(nn.Module):
    """
    Standard ResNet BasicBlock with ECA.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
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
        self.eca = ECA(out_channels)

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
        out = self.eca(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class WideResNetECA(nn.Module):
    """
    Wide ResNet-ECA with Multi-Scale GAP Aggregation.

    Backbone: 3-Stage Wide ResNet (BasicBlock)
    Head: Multi-Scale (Stage 2 + Stage 3) -> GAP -> Concat -> FC
    """

    def __init__(self):
        super(WideResNetECA, self).__init__()
        self.channels = CHANNELS  # [64, 128, 256]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(
                3, self.channels[0], kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True),
        )

        # Stages
        self.stage1 = self._make_layer(self.channels[0], self.channels[0], stride=1)
        self.stage2 = self._make_layer(self.channels[0], self.channels[1], stride=2)
        self.stage3 = self._make_layer(self.channels[1], self.channels[2], stride=2)

        # Final Classifier
        self.fc = nn.Linear(self.channels[1] + self.channels[2], NUM_CLASSES)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, stride, blocks=2):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)

        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        # Multi-Scale Aggregation with standard GAP
        # Stage 2: (B, 128, 16, 16) -> (B, 128)
        feat2 = F.adaptive_avg_pool2d(x2, (1, 1)).flatten(1)
        # Stage 3: (B, 256, 8, 8) -> (B, 256)
        feat3 = F.adaptive_avg_pool2d(x3, (1, 1)).flatten(1)

        combined = torch.cat([feat2, feat3], dim=1)
        logits = self.fc(combined)

        return logits
