import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseProjectedBlock(nn.Module):
    """
    A Residual Block that uses a 3x3 convolution with stride 2 for the shortcut
    connection during downsampling, instead of the standard 1x1 convolution.
    This preserves spatial coherence and improves gradient flow as per the
    "Dense Residual Projections" strategy.
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(DenseProjectedBlock, self).__init__()

        # Main path: 3x3 Conv -> BN -> ReLU -> 3x3 Conv -> BN
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut path
        self.shortcut = nn.Sequential()

        # If dimensions change (stride > 1) or channels change, we need to project
        if stride != 1 or in_channels != self.expansion * out_channels:
            # Structural Innovation: Dense Residual Projection
            # Use 3x3 convolution instead of 1x1 for the shortcut to preserve spatial info
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=3,  # 3x3 kernel
                    stride=stride,
                    padding=1,  # Padding to maintain spatial alignment
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CustomWideResNet(nn.Module):
    """
    Custom Wide ResNet with Dense Residual Projections.

    Architecture:
    - Input: 32x32x3
    - Stem: 3x3 Conv (maintains 32x32)
    - Stage 1: 32 channels (32x32)
    - Stage 2: 64 channels (16x16)
    - Stage 3: 128 channels (8x8)
    - Head: Global Average Pooling -> Linear
    """

    def __init__(self):
        super(CustomWideResNet, self).__init__()

        # Configuration
        channels = Config.MODEL_CHANNELS  # [32, 64, 128]
        num_classes = Config.NUM_CLASSES

        # Initial Stem
        # We start with the number of channels of the first stage
        self.in_channels = channels[0]

        # 3x3 Stem convolution
        self.conv1 = nn.Conv2d(
            Config.CHANNELS,
            self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        # Stages
        # We use 2 blocks per stage as a standard robust configuration for this depth
        # Stage 1: 32 channels, stride 1 -> Output 32x32
        self.layer1 = self._make_layer(
            DenseProjectedBlock, channels[0], blocks=2, stride=1
        )

        # Stage 2: 64 channels, stride 2 -> Output 16x16
        self.layer2 = self._make_layer(
            DenseProjectedBlock, channels[1], blocks=2, stride=2
        )

        # Stage 3: 128 channels, stride 2 -> Output 8x8
        self.layer3 = self._make_layer(
            DenseProjectedBlock, channels[2], blocks=2, stride=2
        )

        # Head: Global Average Pooling -> Linear
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[2], num_classes)

        # Weight Initialization
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []

        # First block handles stride and channel change
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        # Subsequent blocks are identity mappings (stride=1)
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
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
        # Stem
        out = F.relu(self.bn1(self.conv1(x)))

        # Stages
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Head
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
