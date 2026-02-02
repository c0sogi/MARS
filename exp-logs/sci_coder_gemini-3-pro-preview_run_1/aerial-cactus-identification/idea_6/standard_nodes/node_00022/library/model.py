import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicBlock(nn.Module):
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


class CactusResNet(nn.Module):
    """
    Custom Shallow ResNet optimized for 32x32 images (Cite solution_lesson_node_00001).
    Uses conservative downsampling to preserve spatial resolution.
    """

    def __init__(self, num_blocks=[2, 2, 2], num_classes=1):
        super(CactusResNet, self).__init__()
        self.in_planes = 64

        # Stem: 3x3 Conv, Stride 1 (Preserves 32x32 resolution)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 3 Stages
        # Stage 1: 64 channels, stride 1 -> 32x32
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks[0], stride=1)
        # Stage 2: 128 channels, stride 2 -> 16x16
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks[1], stride=2)
        # Stage 3: 256 channels, stride 2 -> 8x8
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks[2], stride=2)

        self.linear = nn.Linear(256 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
