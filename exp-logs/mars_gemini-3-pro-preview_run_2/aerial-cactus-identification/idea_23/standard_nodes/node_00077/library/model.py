import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResNetBlock(nn.Module):
    """
    Standard ResNet Basic Block.
    Optimized for simplicity and stability (Cite Lesson 00008, Lesson 00013).
    """

    def __init__(self, in_channels, out_channels, stride=1, cardinality=1):
        super(ResNetBlock, self).__init__()

        # Groups=1 makes this a standard convolution (ResNet)
        groups = cardinality

        # First 3x3 Conv
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second 3x3 Conv
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection (Cite Lesson 00063: 1x1 Projections)
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

        # Residual connection
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MultiScaleResNet(nn.Module):
    """
    Lightweight Multi-Scale ResNet.
    Optimized for 32x32 images by right-sizing channels and using multi-scale aggregation.
    (Cite Lesson 00016, Lesson 00019)
    """

    def __init__(self):
        super(MultiScaleResNet, self).__init__()

        channels = Config.MODEL_CHANNELS  # [32, 64, 128]
        cardinality = Config.MODEL_CARDINALITY  # 1
        num_classes = Config.NUM_CLASSES

        # Stem: 3x3 convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32x32
        self.stage1 = self._make_layer(
            channels[0], channels[0], stride=1, cardinality=cardinality
        )

        # Stage 2: 16x16
        self.stage2 = self._make_layer(
            channels[0], channels[1], stride=2, cardinality=cardinality
        )

        # Stage 3: 8x8
        self.stage3 = self._make_layer(
            channels[1], channels[2], stride=2, cardinality=cardinality
        )

        # Multi-Scale Head Classifier (Cite Lesson 00016)
        # Input dim = Stage 2 channels + Stage 3 channels
        head_dim = channels[1] + channels[2]
        self.classifier = nn.Linear(head_dim, num_classes)

        self._init_weights()

    def _make_layer(self, in_ch, out_ch, stride, cardinality):
        layers = []
        layers.append(ResNetBlock(in_ch, out_ch, stride, cardinality))
        layers.append(ResNetBlock(out_ch, out_ch, 1, cardinality))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Stage 1
        x = self.stage1(x)

        # Stage 2 (16x16)
        feat_s2 = self.stage2(x)

        # Stage 3 (8x8)
        feat_s3 = self.stage3(feat_s2)

        # Multi-Scale Aggregation Head
        # GAP on Stage 2 features
        pool_s2 = F.adaptive_avg_pool2d(feat_s2, 1).view(x.size(0), -1)

        # GAP on Stage 3 features
        pool_s3 = F.adaptive_avg_pool2d(feat_s3, 1).view(x.size(0), -1)

        # Concatenate features
        combined = torch.cat([pool_s2, pool_s3], dim=1)

        # Final Classification
        logits = self.classifier(combined)

        return logits
