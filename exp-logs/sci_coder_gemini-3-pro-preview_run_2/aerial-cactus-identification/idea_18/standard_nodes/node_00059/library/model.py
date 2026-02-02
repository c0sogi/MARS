import torch
import torch.nn as nn
import torch.nn.functional as F
import library.config as config


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    """
    Standard Residual Block with 2 convolutions.
    Strictly uses 3x3 convolutions for all operations including shortcut projection
    to adhere to the architectural constraints.
    """

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            # Using 3x3 for projection to strictly satisfy "exclusively 3x3 kernels"
            self.shortcut = nn.Sequential(
                conv3x3(inplanes, planes, stride), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class WideResNetPyramidal(nn.Module):
    """
    Custom Wide ResNet with Deeply Supervised Pyramidal Heads.

    Architecture:
    - Input: 32x32
    - Stage 1: 32 channels, 32x32
    - Stage 2: 64 channels, 16x16 (Head A attached here)
    - Stage 3: 128 channels, 8x8 (Head B attached here)
    """

    def __init__(self):
        super(WideResNetPyramidal, self).__init__()

        # Configuration
        self.channels = config.BACKBONE_CHANNELS  # [32, 64, 128]
        self.num_classes = config.NUM_CLASSES

        # Initial Conv (Stem)
        # Mapping 3 input channels to initial width (32)
        self.inplanes = self.channels[0]
        self.conv1 = conv3x3(config.NUM_CHANNELS, self.channels[0])
        self.bn1 = nn.BatchNorm2d(self.channels[0])
        self.relu = nn.ReLU(inplace=True)

        # Stage 1: 32x32 -> 32x32 (32 channels)
        self.layer1 = self._make_layer(BasicBlock, self.channels[0], blocks=2, stride=1)

        # Stage 2: 32x32 -> 16x16 (64 channels)
        self.layer2 = self._make_layer(BasicBlock, self.channels[1], blocks=2, stride=2)

        # Stage 3: 16x16 -> 8x8 (128 channels)
        self.layer3 = self._make_layer(BasicBlock, self.channels[2], blocks=2, stride=2)

        # Head A (Mid-Level): Attached to Stage 2 output (16x16, 64 channels)
        self.head_mid = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.channels[1], self.num_classes),
        )

        # Head B (High-Level): Attached to Stage 3 output (8x8, 128 channels)
        self.head_final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.channels[2], self.num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        # First block handles stride and channel expansion/reduction
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes
        # Subsequent blocks are identity mappings
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Stage 1
        x = self.layer1(x)

        # Stage 2
        x_s2 = self.layer2(x)

        # Stage 3
        x_s3 = self.layer3(x_s2)

        # Deep Supervision Heads
        logits_mid = self.head_mid(x_s2)
        logits_final = self.head_final(x_s3)

        return logits_mid, logits_final
