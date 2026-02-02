import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEModule(nn.Module):
    """
    Standard Squeeze-and-Excitation Module.
    Uses Global Average Pooling (Low-pass filter) and ReLU in the bottleneck
    to enforce sparsity and robustness to speckle noise.
    """

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Ensure at least 1 channel in bottleneck
        reduced_channels = max(1, channels // reduction)

        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),  # Strict ReLU as per design
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y


class ConvBlock(nn.Module):
    """
    Custom Convolutional Block for the Plain CNN Backbone.
    Structure: Conv (Bias=True) -> BN -> LeakyReLU -> SE -> MaxPool.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        # Explicitly retain bias to preserve initialization dynamics
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        # LeakyReLU for spatial layers to preserve negative values (shadows)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.se = SEModule(out_channels, reduction=Config.ATTENTION_REDUCTION_RATIO)
        # Aggressive downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class AGICNN(nn.Module):
    """
    Angle-Gated Isomorphic CNN.
    Features a 4-stage Plain CNN backbone, Selective Hierarchical Readout with
    Isomorphic Pooling, and a Multiplicative Angle-Gating mechanism.
    """

    def __init__(self):
        super(AGICNN, self).__init__()

        # ==========================================
        # 1. Backbone (Plain CNN)
        # ==========================================
        # Input: (B, 3, 75, 75)
        # Channel progression: 3 -> 32 -> 64 -> 128 -> 128
        self.block1 = ConvBlock(Config.CHANNELS, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 128)

        # ==========================================
        # 2. Readout (Corrected Decoupled Isomorphic)
        # ==========================================
        # Decoupled 1x1 Projections for Stage 3 and Stage 4
        self.proj3 = nn.Conv2d(128, 64, kernel_size=1, bias=True)
        self.proj4 = nn.Conv2d(128, 64, kernel_size=1, bias=True)

        # Feature Dimension Calculation:
        # Stage 3: Max(64) + Min(64) = 128
        # Stage 4: Max(64) + Min(64) = 128
        # Total = 256
        self.feature_dim = 256

        # ==========================================
        # 3. Angle-Gating Mechanism
        # ==========================================
        # Maps scalar angle to a channel-wise gain vector G in (0, 1)
        self.angle_gate = nn.Sequential(nn.Linear(1, self.feature_dim), nn.Sigmoid())

        # ==========================================
        # 4. Classification Head
        # ==========================================
        self.dropout = nn.Dropout(p=Config.DROPOUT_RATE)
        self.classifier = nn.Linear(self.feature_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        PyTorch Default Initialization (Kaiming Uniform).
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
                    m.weight, mode="fan_in", nonlinearity="sigmoid"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        """
        Args:
            x: Image tensor (B, 3, 75, 75)
            angle: Incidence angle tensor (B,) or (B, 1)
        """
        # --- Backbone ---
        x = self.block1(x)
        x = self.block2(x)
        x3 = self.block3(x)  # Stage 3 Features
        x4 = self.block4(x3)  # Stage 4 Features

        # --- Decoupled Projections ---
        p3 = self.proj3(x3)  # (B, 64, H3, W3)
        p4 = self.proj4(x4)  # (B, 64, H4, W4)

        # --- Isomorphic Pooling (Max + Min) ---
        # Helper for global min pooling: -Max(-x)

        # Stage 3
        max_p3 = F.adaptive_max_pool2d(p3, 1).view(p3.size(0), -1)
        min_p3 = -F.adaptive_max_pool2d(-p3, 1).view(p3.size(0), -1)
        feat3 = torch.cat([max_p3, min_p3], dim=1)  # 128 dim

        # Stage 4
        max_p4 = F.adaptive_max_pool2d(p4, 1).view(p4.size(0), -1)
        min_p4 = -F.adaptive_max_pool2d(-p4, 1).view(p4.size(0), -1)
        feat4 = torch.cat([max_p4, min_p4], dim=1)  # 128 dim

        # Concatenate to form raw image features
        f_img = torch.cat([feat3, feat4], dim=1)  # 256 dim

        # --- Angle Gating & Calibration ---
        # Ensure angle is (B, 1)
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        # Generate Gain Vector G
        g = self.angle_gate(angle)  # (B, 256)

        # Calibrate: F_cal = F_img * (1 + G)
        # Residual formulation allows default to raw features
        f_calibrated = f_img * (1 + g)

        # --- Classification ---
        out = self.dropout(f_calibrated)
        out = self.classifier(out)

        return out.squeeze(1)  # Return logits (B,)
