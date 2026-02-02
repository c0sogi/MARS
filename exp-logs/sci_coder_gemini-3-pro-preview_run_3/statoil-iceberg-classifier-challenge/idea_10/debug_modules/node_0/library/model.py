import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with Spatial Dropout.
    Structure: Conv -> BN -> ReLU -> Dropout2d -> Conv -> BN
    """

    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.0):
        super(ResidualBlock, self).__init__()

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
        # Spatial Dropout (drops entire channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class MicroResNet(nn.Module):
    """
    Ensembled Robust Micro-ResNet (ERM-ResNet).
    A compact ResNet architecture designed for SAR image classification
    with limited data.

    Key Features:
    - Strict width constraint (max 128 channels).
    - Global Max Pooling to preserve peak signals.
    - Late Fusion with incidence angle.
    - Single hidden layer classification head.
    """

    def __init__(self):
        super(MicroResNet, self).__init__()

        # Configuration
        in_channels = Config.IN_CHANNELS
        base_filters = Config.BASE_FILTERS
        max_filters = Config.MAX_FILTERS
        dropout_spatial = Config.DROPOUT_SPATIAL
        dropout_fc = Config.DROPOUT_FC

        # ---------------------------------------------------------------------
        # Stem
        # ---------------------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )

        # ---------------------------------------------------------------------
        # Backbone (3 Stages)
        # ---------------------------------------------------------------------
        # Stage 1: 64 channels, no stride
        self.stage1 = self._make_layer(
            base_filters, base_filters, stride=1, dropout_prob=dropout_spatial
        )

        # Stage 2: 64 -> 128 channels, stride 2
        self.stage2 = self._make_layer(
            base_filters, max_filters, stride=2, dropout_prob=dropout_spatial
        )

        # Stage 3: 128 -> 128 channels, stride 2
        self.stage3 = self._make_layer(
            max_filters, max_filters, stride=2, dropout_prob=dropout_spatial
        )

        # ---------------------------------------------------------------------
        # Classification Head
        # ---------------------------------------------------------------------
        # Global Max Pooling implies output is 1x1 x Channels
        # Feature vector size = max_filters (128)

        # Late Fusion: 128 features + 1 angle
        self.fc_input_dim = max_filters + 1
        self.hidden_dim = 256  # Single hidden layer size

        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(self.hidden_dim, 1),
        )

        # Weight Initialization
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, stride, dropout_prob):
        """Creates a residual stage with 2 blocks."""
        layers = []
        # First block handles stride and channel expansion
        layers.append(ResidualBlock(in_channels, out_channels, stride, dropout_prob))
        # Second block maintains shape
        layers.append(ResidualBlock(out_channels, out_channels, 1, dropout_prob))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """
        Kaiming Uniform initialization (PyTorch default) for Convs and Linear.
        Explicitly defined to ensure consistency.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image tensor (B, 3, 75, 75)
            angle (torch.Tensor): Incidence angle tensor (B, 1) or (B,)
        """
        # Ensure angle has correct shape (B, 1)
        if angle.dim() == 1:
            angle = angle.unsqueeze(1)

        # Feature Extraction
        out = self.stem(x)  # 64, 75, 75
        out = self.stage1(out)  # 64, 75, 75
        out = self.stage2(out)  # 128, 38, 38
        out = self.stage3(out)  # 128, 19, 19

        # Global Max Pooling
        # Returns (B, C, 1, 1) -> Flatten to (B, C)
        out = F.adaptive_max_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)

        # Late Fusion
        out = torch.cat([out, angle], dim=1)

        # Classification
        logits = self.classifier(out)

        return logits
