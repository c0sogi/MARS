import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BasicBlock(nn.Module):
    """
    Standard Residual Block with two 3x3 convolutions.
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
        # If dimensions change, use 1x1 conv to match dimensions for addition
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


class CustomResNet(nn.Module):
    """
    Lightweight ResNet designed for 32x32 inputs.
    Maintains 8x8 spatial resolution at the final feature map.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(CustomResNet, self).__init__()
        self.in_planes = 32

        # Initial convolution: 3 -> 32 channels. 32x32 output.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Layer 1: 32 channels, stride 1 -> 32x32 output
        self.layer1 = self._make_layer(BasicBlock, 32, 2, stride=1)
        # Layer 2: 64 channels, stride 2 -> 16x16 output
        self.layer2 = self._make_layer(BasicBlock, 64, 2, stride=2)
        # Layer 3: 128 channels, stride 2 -> 8x8 output
        self.layer3 = self._make_layer(BasicBlock, 128, 2, stride=2)

        # Final Classification
        self.linear = nn.Linear(128 * BasicBlock.expansion, num_classes)

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

        # Global Average Pooling: 8x8 -> 1x1
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(_DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(
            in_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = torch.cat([x, out], 1)
        return out


class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class CustomDenseNet(nn.Module):
    """
    Lightweight DenseNet designed for 32x32 inputs.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(CustomDenseNet, self).__init__()
        self.growth_rate = 12

        # Initial Conv
        num_planes = 24
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        # Block 1
        self.block1 = self._make_dense_block(num_planes, self.growth_rate, 4)
        num_planes += 4 * self.growth_rate
        self.trans1 = _Transition(num_planes, num_planes // 2)
        num_planes = num_planes // 2

        # Block 2
        self.block2 = self._make_dense_block(num_planes, self.growth_rate, 4)
        num_planes += 4 * self.growth_rate
        self.trans2 = _Transition(num_planes, num_planes // 2)
        num_planes = num_planes // 2

        # Block 3
        self.block3 = self._make_dense_block(num_planes, self.growth_rate, 4)
        num_planes += 4 * self.growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_block(self, in_channels, growth_rate, n_layers):
        layers = []
        for i in range(n_layers):
            layers.append(_DenseLayer(in_channels + i * growth_rate, growth_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
