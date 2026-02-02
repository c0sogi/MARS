import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    """

    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        # Ensure hidden dimension is at least 1
        hidden_dim = max(in_channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DenseLayer(nn.Module):
    """
    A single layer within a DenseBlock.
    Composite function: BN -> ReLU -> Conv3x3 -> SE -> Dropout.
    """

    def __init__(self, in_channels, growth_rate, drop_rate=0.0, use_se=False):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.drop_rate = drop_rate
        self.use_se = use_se

        if self.use_se:
            self.se = SEBlock(growth_rate)

    def forward(self, x):
        # Standard Pre-activation: BN -> ReLU -> Conv
        out = self.conv(self.relu(self.bn(x)))

        if self.use_se:
            out = self.se(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        return out


class DenseBlock(nn.Module):
    """
    A block of DenseLayers.
    Implements the dense connectivity pattern: the input to each layer is the concatenation
    of the outputs of all preceding layers in the block.
    """

    def __init__(self, num_layers, in_channels, growth_rate, drop_rate, use_se):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels=in_channels + i * growth_rate,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                use_se=use_se,
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x)
            x = torch.cat([x, new_features], 1)
        return x


class TransitionLayer(nn.Module):
    """
    Transition Layer between DenseBlocks.
    Performs downsampling and channel compression.
    BN -> ReLU -> 1x1 Conv -> 2x2 AvgPool.
    """

    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.pool(out)
        return out


class CactusDenseNet(nn.Module):
    """
    Custom DenseNet Architecture for Cactus Identification.
    Features:
    - Specialized Stem (3x3 Conv, stride 1) for 32x32 input.
    - 3 Dense Blocks with Transition Layers.
    - Squeeze-and-Excitation (SE) blocks (optional).
    - Global Average Pooling + Linear Classifier.
    """

    def __init__(
        self,
        growth_rate=Config.GROWTH_RATE,
        block_config=Config.BLOCK_CONFIG,
        compression=Config.COMPRESSION,
        drop_rate=Config.DROP_RATE,
        use_se=Config.USE_SE,
        num_classes=Config.NUM_CLASSES,
    ):
        super(CactusDenseNet, self).__init__()

        # --- Stem ---
        # Standard DenseNet often starts with 2 * growth_rate channels
        num_init_features = 2 * growth_rate
        self.features = nn.Sequential()
        # 3x3 Conv, Stride 1, Padding 1 ensures 32x32 resolution is kept
        self.features.add_module(
            "conv0",
            nn.Conv2d(
                3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False
            ),
        )

        num_features = num_init_features

        # --- Dense Blocks ---
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                use_se=use_se,
            )
            self.features.add_module(f"denseblock{i+1}", block)
            num_features = num_features + num_layers * growth_rate

            # Add Transition Layer if not the last block
            if i != len(block_config) - 1:
                out_features = int(num_features * compression)
                trans = TransitionLayer(num_features, out_features)
                self.features.add_module(f"transition{i+1}", trans)
                num_features = out_features

        # --- Final Batch Norm ---
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # --- Classifier ---
        self.classifier = nn.Linear(num_features, num_classes)

        # --- Weight Initialization ---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
