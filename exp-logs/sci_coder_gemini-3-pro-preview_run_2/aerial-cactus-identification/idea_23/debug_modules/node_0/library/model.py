import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DualPoolingSE(nn.Module):
    """
    Dual-Pooling Squeeze-and-Excitation Block.
    Combines Global Average Pooling and Global Max Pooling to capture
    both global context and salient features (e.g., spines).
    """

    def __init__(self, channels, reduction=16):
        super(DualPoolingSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        # Ensure hidden dimension is at least a reasonable size
        mid_channels = max(channels // reduction, 4)

        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Squeeze: Dual Pooling
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)

        # Aggregate: Summation
        # This combines the signals before the MLP
        y = y_avg + y_max

        # Excitation: Shared MLP
        y = self.fc(y).view(b, c, 1, 1)

        # Scale
        return x * y


class ResNeXtBlock(nn.Module):
    """
    ResNeXt Basic Block with Grouped Convolutions and Dual-Pooling SE.
    Uses 3x3 convolutions exclusively for spatial processing.
    Uses 1x1 convolutions strictly for shortcuts.
    """

    def __init__(
        self, in_channels, out_channels, stride=1, cardinality=32, reduction=16
    ):
        super(ResNeXtBlock, self).__init__()

        # Groups for ResNeXt cardinality
        # Ensure groups divide channels (guaranteed by config [64, 128, 256] and card=32)
        groups = cardinality

        # First 3x3 Grouped Conv
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

        # Second 3x3 Grouped Conv
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

        # Dual-Pooling SE Attention
        self.se = DualPoolingSE(out_channels, reduction)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Strictly utilize 1x1 convolutions for projection shortcuts
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Apply Attention
        out = self.se(out)

        # Residual connection
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class WideResNeXt(nn.Module):
    """
    Custom Wide Dual-Pooling SE-ResNeXt with Multi-Scale Aggregation.

    Architecture:
    - Stem: 3x3 Conv
    - Stage 1: 32x32 (Wide Channels)
    - Stage 2: 16x16 (Wide Channels)
    - Stage 3: 8x8 (Wide Channels)
    - Head: Multi-Scale Aggregation (Stage 2 + Stage 3) -> Classifier
    """

    def __init__(self):
        super(WideResNeXt, self).__init__()

        channels = Config.MODEL_CHANNELS  # [64, 128, 256]
        cardinality = Config.MODEL_CARDINALITY
        num_classes = Config.NUM_CLASSES

        # Stem: 3x3 convolution to preserve 32x32 input resolution initially
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32x32, 64 channels
        self.stage1 = self._make_layer(
            channels[0], channels[0], stride=1, cardinality=cardinality
        )

        # Stage 2: 16x16, 128 channels
        self.stage2 = self._make_layer(
            channels[0], channels[1], stride=2, cardinality=cardinality
        )

        # Stage 3: 8x8, 256 channels
        self.stage3 = self._make_layer(
            channels[1], channels[2], stride=2, cardinality=cardinality
        )

        # Multi-Scale Head Classifier
        # Input dim = Stage 2 channels (128) + Stage 3 channels (256) = 384
        head_dim = channels[1] + channels[2]
        self.classifier = nn.Linear(head_dim, num_classes)

        self._init_weights()

    def _make_layer(self, in_ch, out_ch, stride, cardinality):
        # Using 2 blocks per stage for efficient "Wide" capacity on 32x32 images
        layers = []
        # First block handles potential downsampling/channel expansion
        layers.append(ResNeXtBlock(in_ch, out_ch, stride, cardinality))
        # Second block refines features
        layers.append(ResNeXtBlock(out_ch, out_ch, 1, cardinality))
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
