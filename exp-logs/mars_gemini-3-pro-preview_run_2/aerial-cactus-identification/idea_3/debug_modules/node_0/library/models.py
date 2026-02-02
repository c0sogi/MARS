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


class DenseLayer(nn.Module):
    """
    Single layer within a Dense Block.
    BN -> ReLU -> Conv3x3
    """

    def __init__(self, in_planes, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(
            in_planes, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = torch.cat([x, out], 1)
        return out


class Transition(nn.Module):
    """
    Transition Layer between Dense Blocks.
    BN -> ReLU -> Conv1x1 -> AvgPool2x2
    """

    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class CustomDenseNet(nn.Module):
    """
    Lightweight DenseNet designed for 32x32 inputs.
    Uses 3 Dense Blocks and 2 Transition Layers to reach 8x8 resolution.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(CustomDenseNet, self).__init__()
        self.growth_rate = 12

        # Initial convolution
        num_planes = 24
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        # Block 1: Input 32x32 -> Output 32x32
        self.block1 = self._make_dense_block(num_planes, 4)
        num_planes += 4 * self.growth_rate

        # Transition 1: Input 32x32 -> Output 16x16
        out_planes = int(num_planes * 0.5)
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        # Block 2: Input 16x16 -> Output 16x16
        self.block2 = self._make_dense_block(num_planes, 4)
        num_planes += 4 * self.growth_rate

        # Transition 2: Input 16x16 -> Output 8x8
        out_planes = int(num_planes * 0.5)
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        # Block 3: Input 8x8 -> Output 8x8
        self.block3 = self._make_dense_block(num_planes, 4)
        num_planes += 4 * self.growth_rate

        # Final Batch Norm
        self.bn = nn.BatchNorm2d(num_planes)

        # Linear Layer
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_block(self, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(
                DenseLayer(in_planes + i * self.growth_rate, self.growth_rate)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.trans1(out)
        out = self.block2(out)
        out = self.trans2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out))

        # Global Average Pooling
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
