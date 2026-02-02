import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock(nn.Module):
    """
    A standard Residual Block with two 3x3 convolutions, Batch Normalization, and ReLU.
    Includes a skip connection to facilitate gradient flow.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # First convolution: 3x3, potentially strided
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolution: 3x3, stride 1
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        # If dimensions change (stride > 1) or channels change, apply 1x1 conv to match
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CactusResNet(nn.Module):
    """
    ResNet architecture specifically adapted for 32x32 input images.

    Architecture:
    1. Initial Conv (3x3, stride 1, padding 1) -> 32x32 output
    2. Stage 1: Stack of Residual Blocks (16 channels) -> 32x32 output
    3. Stage 2: Stack of Residual Blocks (32 channels) -> 16x16 output
    4. Stage 3: Stack of Residual Blocks (64 channels) -> 8x8 output
    5. Global Average Pooling -> 1x1 output
    6. Fully Connected Layer -> Class Logits
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, blocks_per_stage=3):
        super(CactusResNet, self).__init__()
        self.in_channels = 16

        # Initial convolution layer
        # Preserves 32x32 spatial dimensions unlike standard ResNet's 7x7 stride 2
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1: 16 channels, 32x32 spatial dim
        self.layer1 = self._make_layer(16, blocks_per_stage, stride=1)

        # Stage 2: 32 channels, 16x16 spatial dim
        self.layer2 = self._make_layer(32, blocks_per_stage, stride=2)

        # Stage 3: 64 channels, 8x8 spatial dim
        self.layer3 = self._make_layer(64, blocks_per_stage, stride=2)

        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final Dense Output Layer
        self.fc = nn.Linear(64, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, out_channels, num_blocks, stride):
        """
        Creates a sequential stack of residual blocks for a stage.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """
        Kaiming initialization for convolutions and normal initialization for linear layers.
        """
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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
