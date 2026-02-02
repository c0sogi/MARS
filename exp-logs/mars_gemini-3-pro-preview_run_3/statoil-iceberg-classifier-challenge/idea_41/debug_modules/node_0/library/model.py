import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block with Global Average Pooling.
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


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv -> BN -> LeakyReLU -> SE -> MaxPool.
    Explicitly retains bias in Conv2d to preserve initialization dynamics.
    """

    def __init__(self, in_channels, out_channels, pool=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True)
        self.se = SEBlock(out_channels)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class ProjectedDualPolarityReadout(nn.Module):
    """
    Projected Dual-Polarity Readout.
    Compresses channels via 1x1 Conv, then extracts both Max and Min pooled features
    to capture signal peaks and shadows without parameter explosion.
    """

    def __init__(self, in_channels, projected_dim):
        super(ProjectedDualPolarityReadout, self).__init__()
        self.projector = nn.Conv2d(in_channels, projected_dim, kernel_size=1, bias=True)

    def forward(self, x):
        # Project: (B, C_in, H, W) -> (B, C_proj, H, W)
        x = self.projector(x)

        # Global Max Pooling: (B, C_proj)
        max_pool = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)

        # Global Min Pooling: (B, C_proj)
        # Implemented as -Max(-x) to capture the most negative values (shadows)
        min_pool = -F.adaptive_max_pool2d(-x, 1).view(x.size(0), -1)

        # Concatenate: (B, C_proj * 2)
        return torch.cat([max_pool, min_pool], dim=1)


class PDPH_SE_CNN(nn.Module):
    """
    Projected Dual-Polarity Hybrid-SE CNN.
    4-Stage Plain CNN Backbone + Dual-Polarity Readout + Angle Fusion.
    """

    def __init__(self):
        super(PDPH_SE_CNN, self).__init__()

        # Backbone Configuration
        channels = Config.BACKBONE_CHANNELS  # [64, 128, 128, 128]

        self.stage1 = ConvBlock(Config.IN_CHANNELS, channels[0])
        self.stage2 = ConvBlock(channels[0], channels[1])
        self.stage3 = ConvBlock(channels[1], channels[2])
        self.stage4 = ConvBlock(channels[2], channels[3])

        # Readout Modules for Stage 3 and Stage 4
        # Input: 128 channels. Output: 64(max) + 64(min) = 128 features.
        self.readout3 = ProjectedDualPolarityReadout(channels[2], Config.PROJECTED_DIM)
        self.readout4 = ProjectedDualPolarityReadout(channels[3], Config.PROJECTED_DIM)

        # Classification Head
        # Inputs: Readout3 (128) + Readout4 (128) + Angle (1) = 257
        input_dim = (Config.PROJECTED_DIM * 2) * 2 + 1
        hidden_dim = 256

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize Weights
        self._init_weights()

    def _init_weights(self):
        """
        Kaiming Uniform Initialization for PyTorch Default consistency.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, angle):
        # Backbone Forward Pass
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        # Dual-Polarity Readout
        r3 = self.readout3(x3)
        r4 = self.readout4(x4)

        # Feature Fusion
        # Ensure angle is (B, 1)
        angle = angle.view(-1, 1)
        features = torch.cat([r3, r4, angle], dim=1)

        # Classification
        out = self.classifier(features)

        # Return shape (B,) for BCEWithLogitsLoss
        return out.squeeze(1)
