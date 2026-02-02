import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel-wise attention.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNeXtBottleneck(nn.Module):
    """
    ResNeXt Bottleneck Block with Grouped Convolutions and SE Attention.
    Strictly uses 1x1 convolutions for projection shortcuts (if downsampling).
    """

    def __init__(
        self, in_channels, out_channels, stride=1, cardinality=32, downsample=None
    ):
        super(ResNeXtBottleneck, self).__init__()

        # 1x1 conv: Projection / Reduction
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 conv: Grouped Convolution (Transform)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv: Expansion / Restoration
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Attention
        self.se = SEBlock(out_channels)

        self.relu = nn.ReLU(inplace=True)
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
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        out += identity
        out = self.relu(out)
        return out


class WideSEResNeXt(nn.Module):
    """
    Custom Wide SE-ResNeXt Architecture.
    Features:
    - High channel capacity [64, 128, 256].
    - 3-Stage backbone preserving 8x8 spatial resolution.
    - Multi-Scale Aggregation Head (Stage 2 + Stage 3).
    """

    def __init__(self, channels=[64, 128, 256], cardinality=32, num_classes=1):
        super(WideSEResNeXt, self).__init__()

        # Stem: 32x32 input -> 32x32 feature map
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32x32 resolution
        self.layer1 = self._make_layer(
            channels[0], channels[0], stride=1, cardinality=cardinality
        )

        # Stage 2: 16x16 resolution
        self.layer2 = self._make_layer(
            channels[0], channels[1], stride=2, cardinality=cardinality
        )

        # Stage 3: 8x8 resolution
        self.layer3 = self._make_layer(
            channels[1], channels[2], stride=2, cardinality=cardinality
        )

        # Head: Multi-Scale Aggregation
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Input dimension is sum of Stage 2 (channels[1]) and Stage 3 (channels[2])
        self.fc = nn.Linear(channels[1] + channels[2], num_classes)

    def _make_layer(self, in_channels, out_channels, stride, cardinality):
        downsample = None
        # Create downsample layer if stride != 1 or channel mismatch
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        return ResNeXtBottleneck(
            in_channels, out_channels, stride, cardinality, downsample
        )

    def forward(self, x):
        # Backbone
        x = self.stem(x)  # 32x32
        x1 = self.layer1(x)  # 32x32
        x2 = self.layer2(x1)  # 16x16
        x3 = self.layer3(x2)  # 8x8

        # Multi-Scale Aggregation
        # Extract features from Stage 2 and Stage 3
        feat2 = self.gap(x2).view(x2.size(0), -1)  # Batch x 128
        feat3 = self.gap(x3).view(x3.size(0), -1)  # Batch x 256

        # Concatenate
        combined = torch.cat([feat2, feat3], dim=1)  # Batch x 384

        # Classification
        logits = self.fc(combined)
        return logits
