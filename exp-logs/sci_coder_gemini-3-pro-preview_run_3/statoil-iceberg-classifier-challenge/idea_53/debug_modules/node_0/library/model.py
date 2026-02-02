import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridSE(nn.Module):
    """
    Squeeze-and-Excitation module using Global Average Pooling.
    """

    def __init__(self, channels, reduction=16):
        super(HybridSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Ensure reduction doesn't make channels too small (min 4)
        reduced_channels = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DIDPCNN(nn.Module):
    """
    Decoupled Isomorphic Dual-Polarity CNN (DIDP-CNN).
    Features a 4-stage Plain CNN backbone with decoupled peak/shadow readouts.
    """

    def __init__(self):
        super(DIDPCNN, self).__init__()

        # Configuration
        self.backbone_channels = Config.BACKBONE_CHANNELS
        in_channels = Config.IN_CHANNELS
        slope = Config.LEAKY_RELU_SLOPE

        # --- Backbone Stages ---
        # Stage 1: 3 -> 64
        self.stage1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.backbone_channels[0],
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(self.backbone_channels[0]),
            nn.LeakyReLU(slope, inplace=True),
            HybridSE(self.backbone_channels[0]),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 2: 64 -> 128
        self.stage2 = nn.Sequential(
            nn.Conv2d(
                self.backbone_channels[0],
                self.backbone_channels[1],
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(self.backbone_channels[1]),
            nn.LeakyReLU(slope, inplace=True),
            HybridSE(self.backbone_channels[1]),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 3: 128 -> 128
        self.stage3 = nn.Sequential(
            nn.Conv2d(
                self.backbone_channels[1],
                self.backbone_channels[2],
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(self.backbone_channels[2]),
            nn.LeakyReLU(slope, inplace=True),
            HybridSE(self.backbone_channels[2]),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 4: 128 -> 128
        self.stage4 = nn.Sequential(
            nn.Conv2d(
                self.backbone_channels[2],
                self.backbone_channels[3],
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(self.backbone_channels[3]),
            nn.LeakyReLU(slope, inplace=True),
            HybridSE(self.backbone_channels[3]),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Decoupled Readout Projections ---
        # Independent subspaces for Peaks and Shadows to prevent feature interference.

        # Stage 3 Readout Projections
        self.s3_peak_conv = nn.Conv2d(
            self.backbone_channels[2], 64, kernel_size=1, bias=True
        )
        self.s3_shadow_conv = nn.Conv2d(
            self.backbone_channels[2], 64, kernel_size=1, bias=True
        )

        # Stage 4 Readout Projections
        self.s4_peak_conv = nn.Conv2d(
            self.backbone_channels[3], 64, kernel_size=1, bias=True
        )
        self.s4_shadow_conv = nn.Conv2d(
            self.backbone_channels[3], 64, kernel_size=1, bias=True
        )

        # --- Classification Head ---
        # Input: 4 vectors * 64 features = 256 + 1 angle = 257
        self.head_fc1 = nn.Linear(256 + 1, 256)
        self.head_drop = nn.Dropout(p=Config.DROPOUT_RATE)
        self.head_fc2 = nn.Linear(256, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Applies Kaiming Uniform initialization to Conv2d and Linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        # Backbone Forward Pass
        x = self.stage1(x)
        x = self.stage2(x)

        # Stage 3
        x3 = self.stage3(x)

        # Stage 4
        x4 = self.stage4(x3)

        # --- Decoupled Readout ---

        # Stage 3 Features
        # Peak: Conv -> Global Max Pool
        s3_p = self.s3_peak_conv(x3)
        s3_p = F.adaptive_max_pool2d(s3_p, (1, 1)).flatten(1)

        # Shadow: Conv -> Global Min Pool (implemented as Max(-x))
        # This captures the magnitude of "dark" features as positive activations
        s3_s = self.s3_shadow_conv(x3)
        s3_s = F.adaptive_max_pool2d(-s3_s, (1, 1)).flatten(1)

        # Stage 4 Features
        # Peak
        s4_p = self.s4_peak_conv(x4)
        s4_p = F.adaptive_max_pool2d(s4_p, (1, 1)).flatten(1)

        # Shadow
        s4_s = self.s4_shadow_conv(x4)
        s4_s = F.adaptive_max_pool2d(-s4_s, (1, 1)).flatten(1)

        # Aggregation: 64*4 = 256
        features = torch.cat([s3_p, s3_s, s4_p, s4_s], dim=1)

        # --- Fusion ---
        # Concatenate scalar angle
        angle = angle.view(-1, 1)
        fused = torch.cat([features, angle], dim=1)

        # --- Head ---
        out = self.head_fc1(fused)
        out = F.leaky_relu(out, negative_slope=Config.LEAKY_RELU_SLOPE)
        out = self.head_drop(out)
        out = self.head_fc2(out)

        return out
