import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Standard Residual Block with 3x3 convolutions.
    """

    expansion = 1

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

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MultiScaleResNet(nn.Module):
    """
    Custom Multi-Scale ResNet for 32x32 images.
    Aggregates features from Stage 2 (16x16) and Stage 3 (8x8) for classification.
    """

    def __init__(self, num_blocks=[2, 2, 2], num_classes=1):
        super(MultiScaleResNet, self).__init__()
        self.in_planes = 16

        # Initial convolution: 3x3, stride 1, padding 1 (maintains 32x32 resolution)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Stage 1: 32x32 -> 32x32
        self.layer1 = self._make_layer(BasicBlock, 16, num_blocks[0], stride=1)
        # Stage 2: 32x32 -> 16x16
        self.layer2 = self._make_layer(BasicBlock, 32, num_blocks[1], stride=2)
        # Stage 3: 16x16 -> 8x8
        self.layer3 = self._make_layer(BasicBlock, 64, num_blocks[2], stride=2)

        # Classification Head
        # We concatenate GAP(Stage 2) [size 32] and GAP(Stage 3) [size 64]
        # Total input features = 32 + 64 = 96
        self.fc = nn.Linear(32 + 64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial Conv
        out = F.relu(self.bn1(self.conv1(x)))

        # Stage 1
        out1 = self.layer1(out)  # Output: [B, 16, 32, 32]

        # Stage 2
        out2 = self.layer2(out1)  # Output: [B, 32, 16, 16]

        # Stage 3
        out3 = self.layer3(out2)  # Output: [B, 64, 8, 8]

        # Multi-Scale Aggregation
        # 1. Global Average Pooling on Stage 2
        gap2 = F.avg_pool2d(out2, out2.size()[2:])  # [B, 32, 1, 1]
        gap2 = gap2.view(gap2.size(0), -1)  # [B, 32]

        # 2. Global Average Pooling on Stage 3
        gap3 = F.avg_pool2d(out3, out3.size()[2:])  # [B, 64, 1, 1]
        gap3 = gap3.view(gap3.size(0), -1)  # [B, 64]

        # 3. Concatenate features
        combined = torch.cat([gap2, gap3], dim=1)  # [B, 96]

        # Classification
        out = self.fc(combined)
        return out
