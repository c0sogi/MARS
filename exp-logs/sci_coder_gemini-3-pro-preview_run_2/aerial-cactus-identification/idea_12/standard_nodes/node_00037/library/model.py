import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import MODEL_CHANNELS


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    """

    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    """
    ResNet Basic Block with Squeeze-and-Excitation.
    """

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.se = SEBlock(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class NarrowSEResNet(nn.Module):
    """
    Custom Narrow SE-ResNet with Multi-Scale Global Average Pooling.
    """

    def __init__(self):
        super(NarrowSEResNet, self).__init__()
        # Channels: [16, 32, 64]
        c1, c2, c3 = MODEL_CHANNELS

        # Initial Conv: 3->16, 32x32 output (no downsampling yet)
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1: 16 channels, 32x32
        self.layer1 = self._make_layer(c1, c1, stride=1, num_blocks=2)

        # Stage 2: 32 channels, 16x16 (stride 2)
        self.layer2 = self._make_layer(c1, c2, stride=2, num_blocks=2)

        # Stage 3: 64 channels, 8x8 (stride 2)
        self.layer3 = self._make_layer(c2, c3, stride=2, num_blocks=2)

        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        # Concatenating features from Stage 2 and Stage 3
        self.fc = nn.Linear(c2 + c3, 1)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, planes, stride, num_blocks):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input: (B, 3, 32, 32)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Stage 1: (B, 16, 32, 32)
        out = self.layer1(out)

        # Stage 2: (B, 32, 16, 16)
        out2 = self.layer2(out)

        # Stage 3: (B, 64, 8, 8)
        out3 = self.layer3(out2)

        # Multi-scale aggregation with GAP
        # Flatten after pooling: (B, C, 1, 1) -> (B, C)
        feat2 = self.avg_pool(out2).view(out2.size(0), -1)
        feat3 = self.avg_pool(out3).view(out3.size(0), -1)

        # Concatenate features: (B, 32+64)
        combined = torch.cat((feat2, feat3), dim=1)

        # Classification
        logits = self.fc(combined)
        return logits
