import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block.
    Uses 3x3 convolutions and 1x1 projection shortcuts.
    Cite solution_lesson_node_00063: Prefer 1x1 convolutions for projection shortcuts.
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
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
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class NarrowResNetMSA(nn.Module):
    """
    Narrow ResNet with Multi-Scale Aggregation.
    Cite solution_lesson_node_00019: Scale channel width proportionally to input resolution (Narrow [16, 32, 64]).
    Cite solution_lesson_node_00016: Efficiency via Multi-Scale Feature Aggregation.
    Cite solution_lesson_node_00013: Avoid architectural overhead (removed SE blocks).
    """

    def __init__(self, channels=[16, 32, 64], num_classes=1):
        super(NarrowResNetMSA, self).__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32x32
        self.layer1 = self._make_layer(channels[0], channels[0], stride=1)
        # Stage 2: 16x16
        self.layer2 = self._make_layer(channels[0], channels[1], stride=2)
        # Stage 3: 8x8
        self.layer3 = self._make_layer(channels[1], channels[2], stride=2)

        # Head: Multi-Scale Aggregation
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[1] + channels[2], num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        return BasicBlock(in_channels, out_channels, stride, downsample)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        # Multi-Scale Aggregation: Stage 2 + Stage 3
        feat2 = self.gap(x2).view(x2.size(0), -1)
        feat3 = self.gap(x3).view(x3.size(0), -1)

        combined = torch.cat([feat2, feat3], dim=1)
        logits = self.fc(combined)
        return logits
