import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    """
    Residual Block with Squeeze-and-Excitation.
    Supports downsampling via stride and channel projection.
    """

    def __init__(
        self, in_channels, out_channels, stride=1, use_se=True, se_reduction=16
    ):
        super(ResBlock, self).__init__()
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

        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(out_channels, se_reduction)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class WideSEResNet(nn.Module):
    """
    Custom 4-Stage Residual Network with Aggressive Downsampling and Squeeze-and-Excitation.
    Designed for 75x75 SAR images, reducing them to ~5x5 before pooling.
    """

    def __init__(self):
        super(WideSEResNet, self).__init__()

        # Configuration
        widths = Config.CHANNEL_WIDTHS  # Expected: [64, 128, 128, 128]
        use_se = Config.USE_SE
        se_reduction = Config.SE_REDUCTION
        dropout_rate = Config.DROPOUT_RATE
        num_classes = Config.NUM_CLASSES

        # Initial Stem
        # 75x75 -> 75x75 (Stride 1)
        self.stem = nn.Sequential(
            nn.Conv2d(
                Config.IN_CHANNELS,
                widths[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 75x75 -> 38x38 (Stride 2) - Width: 64
        self.stage1 = ResBlock(
            widths[0], widths[0], stride=2, use_se=use_se, se_reduction=se_reduction
        )

        # Stage 2: 38x38 -> 19x19 (Stride 2) - Expansion to 128 channels
        self.stage2 = ResBlock(
            widths[0], widths[1], stride=2, use_se=use_se, se_reduction=se_reduction
        )

        # Stage 3: 19x19 -> 10x10 (Stride 2) - Width: 128
        self.stage3 = ResBlock(
            widths[1], widths[2], stride=2, use_se=use_se, se_reduction=se_reduction
        )

        # Stage 4: 10x10 -> 5x5 (Stride 2) - Width: 128
        self.stage4 = ResBlock(
            widths[2], widths[3], stride=2, use_se=use_se, se_reduction=se_reduction
        )

        # Global Max Pooling
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        # Classification Head
        # Input: Flattened Features (128) + Incidence Angle (1) = 129
        # Structure: Linear -> ReLU -> Dropout -> Output
        input_dim = widths[3] + 1
        hidden_dim = 256  # Intermediate hidden dimension

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        PyTorch Default Initialization (Kaiming Uniform)
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
        # Feature Extraction
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten (B, 128)

        # Feature Fusion
        # Concatenate raw incidence angle
        angle = angle.view(-1, 1)  # Ensure (B, 1)
        x = torch.cat([x, angle], dim=1)  # (B, 129)

        # Classification
        x = self.head(x)

        return x.squeeze(1)  # Return (B)
