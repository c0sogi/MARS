import torch
import torch.nn as nn
import torch.nn.functional as F
from library.coordinate_attention import CoordinateAttention


class ResNeXtBlock(nn.Module):
    """
    ResNeXt Block with Coordinate Attention.

    Standard ResNeXt bottleneck block (1x1 -> 3x3 Grouped -> 1x1) enhanced with
    Coordinate Attention to preserve spatial information.
    """

    def __init__(self, in_planes, planes, stride=1, cardinality=32, reduction=16):
        super(ResNeXtBlock, self).__init__()

        # In Wide ResNeXt, the bottleneck width is typically planes // 2
        group_width = planes // 2

        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)

        self.conv2 = nn.Conv2d(
            group_width,
            group_width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(group_width)

        self.conv3 = nn.Conv2d(group_width, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        # Coordinate Attention replaces SE blocks
        self.attn = CoordinateAttention(planes, reduction=reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # Apply Coordinate Attention
        out = self.attn(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class WideCoordinateResNeXt(nn.Module):
    """
    Wide Coordinate-ResNeXt with Multi-Scale Aggregation.

    Backbone: 3-stage ResNeXt with [64, 128, 256] channels.
    Head: Concatenates GAP features from Stage 2 (16x16) and Stage 3 (8x8).
    """

    def __init__(self, cardinality=32):
        super(WideCoordinateResNeXt, self).__init__()
        self.cardinality = cardinality

        # Initial Conv: 32x32 input
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Stage 1: 32x32 -> 32x32. Channels: 64
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)

        # Stage 2: 32x32 -> 16x16. Channels: 128
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)

        # Stage 3: 16x16 -> 8x8. Channels: 256
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        # Head: Multi-Scale Aggregation
        # Concatenating GAP of Stage 2 (128 ch) and Stage 3 (256 ch)
        self.fc = nn.Linear(128 + 256, 1)

    def _make_layer(self, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(
                ResNeXtBlock(in_planes, planes, stride=s, cardinality=self.cardinality)
            )
            in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial Conv
        out = F.relu(self.bn1(self.conv1(x)))  # 32x32

        # Backbone Stages
        out1 = self.layer1(out)  # 32x32
        out2 = self.layer2(out1)  # 16x16
        out3 = self.layer3(out2)  # 8x8

        # Multi-Scale Aggregation
        # GAP on Stage 2 (Mid-level features)
        gap2 = F.adaptive_avg_pool2d(out2, 1).view(out2.size(0), -1)  # 128

        # GAP on Stage 3 (High-level features)
        gap3 = F.adaptive_avg_pool2d(out3, 1).view(out3.size(0), -1)  # 256

        # Concatenate and Classify
        combined = torch.cat([gap2, gap3], dim=1)
        return self.fc(combined)
