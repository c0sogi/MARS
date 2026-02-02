import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseLayer(nn.Module):
    """
    A single layer within a DenseBlock.
    Structure: BN -> ReLU -> Conv3x3 -> Dropout (optional)
    """

    def __init__(self, in_channels, growth_rate, drop_rate=0.0):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(
            in_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    """
    A block consisting of multiple DenseLayers.
    """

    def __init__(self, in_channels, growth_rate, num_layers, drop_rate=0.0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                DenseLayer(in_channels + i * growth_rate, growth_rate, drop_rate)
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class TransitionLayer(nn.Module):
    """
    Transition layer to reduce feature map size and channel count between blocks.
    Structure: BN -> ReLU -> Conv1x1 -> AvgPool2x2
    """

    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class CustomDenseNet(nn.Module):
    """
    Custom DenseNet architecture for 75x75 radar images.
    Features:
    - 3 Dense Blocks
    - Global Max Pooling (to capture high intensity peaks)
    - Incidence Angle Fusion
    """

    def __init__(
        self,
        growth_rate=12,
        block_config=(6, 12, 16),
        num_init_features=24,
        drop_rate=Config.DROP_RATE,
        fc_dim=Config.FC_DIM,
    ):
        super(CustomDenseNet, self).__init__()

        # 1. Initial Convolution
        # Input: (N, 3, 75, 75)
        self.features = nn.Sequential(
            nn.Conv2d(
                Config.IN_CHANNELS,
                num_init_features,
                kernel_size=3,
                padding=1,
                bias=False,
            )
        )

        # 2. Dense Blocks and Transition Layers
        self.blocks = nn.ModuleList()
        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_features, growth_rate, num_layers, drop_rate)
            self.blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            # Add transition layer if not the last block
            if i != len(block_config) - 1:
                out_features = num_features // 2
                trans = TransitionLayer(num_features, out_features)
                self.blocks.append(trans)
                num_features = out_features

        # Final Batch Norm before pooling
        self.final_bn = nn.BatchNorm2d(num_features)

        # 3. Classification Head
        # Input dim = num_features (from image) + 1 (incidence angle)
        self.classifier = nn.Sequential(
            nn.Linear(num_features + 1, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x, angle):
        # Feature Extraction
        out = self.features(x)

        for block in self.blocks:
            out = block(out)

        out = F.relu(self.final_bn(out))

        # Global Max Pooling
        # Reduces (N, C, H, W) -> (N, C)
        out = F.max_pool2d(out, kernel_size=out.size()[2:]).view(out.size(0), -1)

        # Feature Fusion
        # Concatenate image features with incidence angle
        angle = angle.view(-1, 1)  # Ensure shape (N, 1)
        out = torch.cat([out, angle], dim=1)

        # Classification
        out = self.classifier(out)

        return out
