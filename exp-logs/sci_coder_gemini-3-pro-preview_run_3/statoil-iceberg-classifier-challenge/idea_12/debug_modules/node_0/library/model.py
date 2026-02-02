import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with Squeeze-and-Excitation.
    """

    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
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

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels, reduction)

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

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        out += residual
        out = self.relu(out)
        return out


class IcebergSEResNet(nn.Module):
    """
    4-Stage Residual Network with SE Attention and Aggressive Downsampling.
    Includes Late Fusion with Incidence Angle.
    """

    def __init__(self):
        super(IcebergSEResNet, self).__init__()

        # Load configuration
        self.in_channels = Config.IN_CHANNELS
        self.stem_channels = Config.STEM_CHANNELS
        self.stage_channels = Config.STAGE_CHANNELS
        self.se_reduction = Config.SE_REDUCTION
        self.head_dropout = Config.HEAD_DROPOUT

        # Initial Stem: 75x75 -> 75x75 (Stride 1)
        self.stem = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.stem_channels),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 64 -> 64, Stride 2 (75 -> 38)
        self.stage1 = self._make_layer(
            self.stem_channels, self.stage_channels[0], stride=2
        )

        # Stage 2: 64 -> 128, Stride 2 (38 -> 19) - Early Expansion
        self.stage2 = self._make_layer(
            self.stage_channels[0], self.stage_channels[1], stride=2
        )

        # Stage 3: 128 -> 128, Stride 2 (19 -> 10)
        self.stage3 = self._make_layer(
            self.stage_channels[1], self.stage_channels[2], stride=2
        )

        # Stage 4: 128 -> 128, Stride 2 (10 -> 5)
        self.stage4 = self._make_layer(
            self.stage_channels[2], self.stage_channels[3], stride=2
        )

        # Global Max Pooling: 5x5 -> 1x1
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        # Classification Head with Late Fusion
        # Concatenate flattened features (128) + incidence angle (1)
        cnn_out_dim = self.stage_channels[3]
        total_in_features = cnn_out_dim + 1

        # Hidden dimension for the head (matching max width of CNN)
        hidden_dim = 128

        self.head = nn.Sequential(
            nn.Linear(total_in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.head_dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, stride):
        return ResidualBlock(
            in_channels, out_channels, stride=stride, reduction=self.se_reduction
        )

    def _initialize_weights(self):
        # Kaiming Uniform Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image input [Batch, 3, 75, 75]
            angle (torch.Tensor): Incidence angle input [Batch]
        """
        # Feature Extraction
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Late Fusion
        angle = angle.view(-1, 1)  # Ensure shape [Batch, 1]
        x = torch.cat([x, angle], dim=1)

        # Classification
        x = self.head(x)

        return x
