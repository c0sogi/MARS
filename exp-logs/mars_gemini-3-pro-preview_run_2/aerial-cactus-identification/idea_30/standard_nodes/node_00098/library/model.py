import torch
import torch.nn as nn
import torch.nn.functional as F
from library.layers import ECA, BlurPool, GeM
from library.config import CHANNELS, CARDINALITY, NUM_CLASSES


class ResNeXtBottleneck(nn.Module):
    """
    Wide ResNeXt Bottleneck with ECA and Anti-Aliased Downsampling.

    Structure:
    - 1x1 Conv (Projection to hidden_dim)
    - 3x3 Grouped Conv (Stride 1)
    - BlurPool (Downsampling if stride > 1)
    - 1x1 Conv (Expansion to out_channels)
    - ECA Attention
    - Shortcut (with BlurPool if downsampling needed)
    """

    def __init__(self, in_channels, out_channels, stride=1, cardinality=32):
        super(ResNeXtBottleneck, self).__init__()

        # "Wide" configuration: We set hidden_dim equal to out_channels (expansion=1)
        # effectively making the bottleneck very wide compared to standard ResNeXt (expansion=2 or 4).
        hidden_dim = out_channels
        self.stride = stride

        # 1. Reduce / Project
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        # 2. 3x3 Grouped Convolution
        # Always Stride 1 to preserve features before anti-aliased downsampling
        self.conv2 = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        # 3. Anti-Aliased Downsampling (BlurPool)
        # Applied after the 3x3 convolution if stride > 1
        self.blur_pool = (
            BlurPool(hidden_dim, stride=stride) if stride > 1 else nn.Identity()
        )

        # 4. Expand
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # 5. Attention (ECA)
        self.eca = ECA(out_channels)

        # 6. Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Strictly utilize 1x1 convolutions for projection
            # To maintain anti-aliasing: 1x1 Conv (Stride 1) -> BlurPool (Stride s)
            shortcut_layers = []
            shortcut_layers.append(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                )
            )
            shortcut_layers.append(nn.BatchNorm2d(out_channels))
            if stride > 1:
                shortcut_layers.append(BlurPool(out_channels, stride=stride))
            self.shortcut = nn.Sequential(*shortcut_layers)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Apply BlurPool
        if self.stride > 1:
            out = self.blur_pool(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply ECA
        out = self.eca(out)

        out += residual
        out = self.relu(out)

        return out


class CustomWideResNeXtECA(nn.Module):
    """
    Custom Wide Anti-Aliased ResNeXt-ECA with Multi-Scale GeM Aggregation.

    Backbone: 3-Stage Wide ResNeXt
    Attention: ECA
    Downsampling: BlurPool
    Head: Multi-Scale (Stage 2 + Stage 3) -> GeM -> Concat -> FC
    """

    def __init__(self):
        super(CustomWideResNeXtECA, self).__init__()

        self.cardinality = CARDINALITY
        self.channels = CHANNELS  # [64, 128, 256]

        # Stem: 3x3 Conv -> BN -> ReLU
        # Input 32x32
        self.stem = nn.Sequential(
            nn.Conv2d(
                3, self.channels[0], kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 64 -> 64, Stride 1 (32x32)
        self.stage1 = self._make_layer(self.channels[0], self.channels[0], stride=1)

        # Stage 2: 64 -> 128, Stride 2 (32x32 -> 16x16)
        self.stage2 = self._make_layer(self.channels[0], self.channels[1], stride=2)

        # Stage 3: 128 -> 256, Stride 2 (16x16 -> 8x8)
        self.stage3 = self._make_layer(self.channels[1], self.channels[2], stride=2)

        # Multi-Scale GeM Aggregation Head
        self.gem2 = GeM()
        self.gem3 = GeM()

        # Final Classifier
        # Input features: Stage 2 channels + Stage 3 channels
        self.fc = nn.Linear(self.channels[1] + self.channels[2], NUM_CLASSES)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, stride, blocks=2):
        layers = []
        # First block handles stride and potential channel expansion
        layers.append(
            ResNeXtBottleneck(in_channels, out_channels, stride, self.cardinality)
        )
        # Subsequent blocks are identity mappings
        for _ in range(1, blocks):
            layers.append(
                ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality)
            )
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
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Backbone Stages
        x1 = self.stage1(x)  # (B, 64, 32, 32)
        x2 = self.stage2(x1)  # (B, 128, 16, 16)
        x3 = self.stage3(x2)  # (B, 256, 8, 8)

        # Multi-Scale Aggregation
        # Extract features from Stage 2 and Stage 3
        feat2 = self.gem2(x2)  # (B, 128)
        feat3 = self.gem3(x3)  # (B, 256)

        # Concatenate
        combined = torch.cat([feat2, feat3], dim=1)  # (B, 384)

        # Classification
        logits = self.fc(combined)

        return logits
