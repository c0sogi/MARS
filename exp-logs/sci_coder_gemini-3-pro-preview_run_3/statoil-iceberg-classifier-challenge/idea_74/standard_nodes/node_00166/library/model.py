import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEModule(nn.Module):
    """
    Standard Squeeze-and-Excitation Module.
    """

    def __init__(self, channels, reduction=Config.SE_Reduction):
        super(SEModule, self).__init__()
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
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """
    Convolutional Block: Conv2d -> BN -> LeakyReLU -> SE -> MaxPool
    Explicitly retains bias in Conv2d as per design.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=Config.Use_Bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=Config.Negative_Slope, inplace=True)
        self.se = SEModule(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class FiLMCalibrator(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Calibrator.
    Modulates feature vectors based on incidence angle.
    """

    def __init__(self, feature_dim, hidden_dim=Config.Calibration_Dim):
        super(FiLMCalibrator, self).__init__()
        self.feature_dim = feature_dim

        # Angle Encoder
        self.angle_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(negative_slope=Config.Negative_Slope, inplace=True),
        )

        # Projector for Scale (gamma) and Shift (beta)
        # Output size is 2 * feature_dim
        self.projector = nn.Linear(hidden_dim, 2 * feature_dim)

    def forward(self, feature_vector, angle):
        """
        Args:
            feature_vector: Tensor of shape (B, feature_dim)
            angle: Tensor of shape (B, ) or (B, 1)
        """
        # Ensure angle is (B, 1)
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        # Encode angle
        encoded = self.angle_encoder(angle)

        # Get params
        params = self.projector(encoded)  # (B, 2 * feature_dim)

        # Split into gamma and beta
        gamma, beta = torch.split(params, self.feature_dim, dim=1)

        # Modulation: V' = V * Sigmoid(gamma) + beta
        # Using Sigmoid for gamma ensures scale is in (0, 1), acting as a gate/gain
        # Adding beta allows shifting the distribution
        modulated = feature_vector * torch.sigmoid(gamma) + beta

        return modulated


class HCICNN(nn.Module):
    """
    Hierarchically Calibrated Isomorphic CNN.

    Structure:
    - 4-Stage Plain CNN Backbone
    - Decoupled Projections for Stage 3 and 4
    - Isomorphic Dual-Polarity Pooling (Max+Min)
    - Hierarchical FiLM Calibration using Incidence Angle
    - Fusion and Classification Head
    """

    def __init__(self):
        super(HCICNN, self).__init__()

        # --- Backbone ---
        # Stage 1: 3 -> 64
        self.block1 = ConvBlock(Config.IN_CHANNELS, 64)
        # Stage 2: 64 -> 128
        self.block2 = ConvBlock(64, 128)
        # Stage 3: 128 -> 128
        self.block3 = ConvBlock(128, 128)
        # Stage 4: 128 -> 128
        self.block4 = ConvBlock(128, 128)

        # --- Readout (Decoupled Isomorphic) ---
        # We want final feature vector per stage to be 128 dims.
        # Dual polarity (Max+Min) doubles the dimension, so we project to 64 first.
        # 64 (Max) + 64 (Min) = 128.

        self.proj3 = nn.Conv2d(128, 64, kernel_size=1)
        self.proj4 = nn.Conv2d(128, 64, kernel_size=1)

        # --- Calibration ---
        # Feature_Dim is 128 (from Config)
        self.calib3 = FiLMCalibrator(feature_dim=128)
        self.calib4 = FiLMCalibrator(feature_dim=128)

        # --- Head ---
        # Concatenation of V3' and V4' -> 128 + 128 = 256
        self.dropout = nn.Dropout(p=Config.Dropout_Rate)
        self.classifier = nn.Linear(256, 1)

        # --- Initialization ---
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _global_pool(self, x):
        """
        Isomorphic Dual-Polarity Pooling: Global Max + Global Min
        """
        # x shape: (B, C, H, W)
        # Max Pooling
        max_pool = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
        # Min Pooling: -max(-x)
        min_pool = -F.adaptive_max_pool2d(-x, 1).view(x.size(0), -1)

        # Concatenate: (B, 2*C)
        return torch.cat([max_pool, min_pool], dim=1)

    def forward(self, x, angle):
        # --- Backbone ---
        x = self.block1(x)
        x = self.block2(x)

        # Stage 3
        x3 = self.block3(x)

        # Stage 4
        x4 = self.block4(x3)

        # --- Readout ---
        # Process Stage 3 features
        p3 = self.proj3(x3)
        v3 = self._global_pool(p3)  # (B, 128)

        # Process Stage 4 features
        p4 = self.proj4(x4)
        v4 = self._global_pool(p4)  # (B, 128)

        # --- Hierarchical Calibration ---
        v3_calib = self.calib3(v3, angle)
        v4_calib = self.calib4(v4, angle)

        # --- Fusion ---
        v_final = torch.cat([v3_calib, v4_calib], dim=1)  # (B, 256)

        # --- Classification ---
        out = self.dropout(v_final)
        out = self.classifier(out)

        return out
