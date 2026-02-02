import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel-wise attention.
    """

    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):
    """
    Standard Residual Block with optional SE module.
    """

    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(ResBlock, self).__init__()
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

        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_channels, reduction=Config.SE_REDUCTION)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class NarrowMultiScaleResNet(nn.Module):
    """
    Custom Narrow Multi-Scale ResNet with Global Average Pooling.
    Aggregates features from Stage 2 and Stage 3 (Cite solution_lesson_node_00016).
    """

    def __init__(self):
        super(NarrowMultiScaleResNet, self).__init__()

        channels = Config.BLOCK_CHANNELS  # Expected: [16, 32, 64]

        # Initial Convolution: 3 -> 16, 32x32
        self.conv1 = nn.Conv2d(
            Config.IN_CHANNELS,
            channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)

        # Stage 1: 16 channels, 32x32 (No downsampling)
        self.layer1 = self._make_layer(channels[0], channels[0], stride=1)

        # Stage 2: 32 channels, 16x16 (Downsampling)
        self.layer2 = self._make_layer(channels[0], channels[1], stride=2)

        # Stage 3: 64 channels, 8x8 (Downsampling)
        self.layer3 = self._make_layer(channels[1], channels[2], stride=2)

        # Multi-Scale Aggregation: Concatenate GAP from Stage 2 and Stage 3
        total_dim = channels[1] + channels[2]  # 32 + 64 = 96

        self.fc = nn.Linear(total_dim, 1)

        # Weight initialization
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

    def _make_layer(self, in_channels, out_channels, stride, blocks=2):
        layers = []
        layers.append(
            ResBlock(in_channels, out_channels, stride=stride, use_se=Config.USE_SE)
        )
        for _ in range(1, blocks):
            layers.append(
                ResBlock(out_channels, out_channels, stride=1, use_se=Config.USE_SE)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial Conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Stage 1
        x = self.layer1(x)  # Output: (B, 16, 32, 32)

        # Stage 2
        f2 = self.layer2(x)  # Output: (B, 32, 16, 16)

        # Stage 3
        f3 = self.layer3(f2)  # Output: (B, 64, 8, 8)

        # Multi-Scale Global Average Pooling
        p2 = F.adaptive_avg_pool2d(f2, (1, 1)).flatten(1)  # (B, 32)
        p3 = F.adaptive_avg_pool2d(f3, (1, 1)).flatten(1)  # (B, 64)

        # Feature Fusion
        out = torch.cat([p2, p3], dim=1)  # Size: B x 96

        # Classification
        out = self.fc(out)

        return out
