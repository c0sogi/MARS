import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResNetBlock(nn.Module):
    """
    Standard ResNet Block with 3x3 convolutions and a residual connection.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
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

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class NarrowResNetEncoder(nn.Module):
    """
    Narrow ResNet Encoder with channel configuration [16, 32, 64].
    Designed for 32x32 input images, avoiding aggressive initial downsampling.
    """

    def __init__(self):
        super(NarrowResNetEncoder, self).__init__()
        channels = Config.ENCODER_CHANNELS  # Expected: [16, 32, 64]

        # Initial convolution: 3 -> 16. Stride 1 to preserve 32x32 resolution.
        self.init_conv = nn.Conv2d(
            3, channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.init_bn = nn.BatchNorm2d(channels[0])
        self.init_relu = nn.ReLU(inplace=True)

        # Stage 1: 16 channels, 32x32 resolution
        self.layer1 = ResNetBlock(channels[0], channels[0], stride=1)

        # Stage 2: 32 channels, 16x16 resolution
        self.layer2 = ResNetBlock(channels[0], channels[1], stride=2)

        # Stage 3: 64 channels, 8x8 resolution (Bottleneck)
        self.layer3 = ResNetBlock(channels[1], channels[2], stride=2)

    def forward(self, x):
        # Initial processing
        x = self.init_relu(self.init_bn(self.init_conv(x)))

        # Encoder stages with skip connection extraction
        f1 = self.layer1(x)  # 32x32, 16ch
        f2 = self.layer2(f1)  # 16x16, 32ch
        f3 = self.layer3(f2)  # 8x8, 64ch (Bottleneck)

        return [f1, f2, f3]


class MultiScaleNarrowResNet(nn.Module):
    """
    Multi-Scale Narrow ResNet.
    Aggregates features from Stage 2 (16x16) and Stage 3 (8x8) of the encoder.
    Cite solution_lesson_node_00016 (Multi-Scale Aggregation)
    Cite solution_lesson_node_00027 (Avoid U-Net)
    """

    def __init__(self):
        super(MultiScaleNarrowResNet, self).__init__()
        self.encoder = NarrowResNetEncoder()
        # Encoder channels: [16, 32, 64]
        # Stage 2 output (f2): 32 channels
        # Stage 3 output (f3): 64 channels
        self.classifier = nn.Linear(32 + 64, Config.NUM_CLASSES)

    def forward(self, x):
        features = self.encoder(x)
        # features = [f1, f2, f3]
        f2 = features[1]
        f3 = features[2]

        # Global Average Pooling
        f2_pooled = F.adaptive_avg_pool2d(f2, (1, 1)).flatten(1)
        f3_pooled = F.adaptive_avg_pool2d(f3, (1, 1)).flatten(1)

        # Concatenate
        combined = torch.cat([f2_pooled, f3_pooled], dim=1)

        return self.classifier(combined)
