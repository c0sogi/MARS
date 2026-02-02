import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MADSELayer(nn.Module):
    """
    Mean Absolute Deviation (MAD) Squeeze-and-Excitation Module.

    Computes two statistics per channel:
    1. Global Mean (mu)
    2. Global Mean Absolute Deviation (delta = Avg(|x - mu|))

    This provides a robust measure of texture/dispersion that is linear with respect to
    pixel intensity, avoiding the amplification of speckle noise inherent in
    variance/standard deviation calculations.
    """

    def __init__(self, channels, reduction=16):
        super(MADSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Input to MLP is 2 * channels (Mean + MAD)
        input_dim = 2 * channels
        reduced_dim = max(1, channels // reduction)

        # MLP structure: Linear -> ReLU -> Linear -> Sigmoid
        # We strictly use ReLU in the bottleneck as per solution constraints.
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # 1. Global Mean: mu_c = Avg(x)
        # Output shape: (B, C, 1, 1)
        mu = self.avg_pool(x)

        # 2. Global Mean Absolute Deviation: delta_c = Avg(|x - mu_c|)
        # Output shape: (B, C, 1, 1)
        mad = (x - mu).abs().mean(dim=(2, 3), keepdim=True)

        # Concatenate statistics along channel dimension
        # Shape: (B, 2C)
        stats = torch.cat([mu.view(b, -1), mad.view(b, -1)], dim=1)

        # Excitation: Learn channel-wise weights
        # Shape: (B, C, 1, 1)
        scale = self.fc(stats).view(b, c, 1, 1)

        # Scale input
        return x * scale


class RTICNN(nn.Module):
    """
    Robust-Texture Isomorphic CNN (RTI-CNN).

    Architecture:
    - Backbone: 4-Stage Plain CNN (no residuals) to enforce downsampling of noise.
    - Regularization: MAD-SE modules in every block.
    - Activation: LeakyReLU (0.1) to preserve shadow information (negative values).
    - Readout: Corrected Decoupled Isomorphic (Separate Projections + Max/Min Pooling).
    - Fusion: Concatenation with raw incidence angle.
    """

    def __init__(self):
        super(RTICNN, self).__init__()

        # Retrieve configuration
        in_channels = Config.INPUT_CHANNELS
        widths = Config.BACKBONE_CHANNELS  # [64, 128, 128, 128]
        slope = Config.LEAKY_RELU_SLOPE
        se_reduction = Config.SE_REDUCTION
        proj_dim = Config.PROJECTION_DIM
        feature_dim = Config.FEATURE_DIM
        dropout_rate = Config.DROPOUT_RATE

        # --- Backbone (4 Stages) ---
        # We explicitly retain bias=True in Conv2d layers.

        # Stage 1: 75x75 -> 37x37
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(widths[0]),
            nn.LeakyReLU(slope, inplace=True),
            MADSELayer(widths[0], reduction=se_reduction),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 2: 37x37 -> 18x18
        self.layer2 = nn.Sequential(
            nn.Conv2d(widths[0], widths[1], kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(widths[1]),
            nn.LeakyReLU(slope, inplace=True),
            MADSELayer(widths[1], reduction=se_reduction),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 3: 18x18 -> 9x9
        self.layer3 = nn.Sequential(
            nn.Conv2d(widths[1], widths[2], kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(widths[2]),
            nn.LeakyReLU(slope, inplace=True),
            MADSELayer(widths[2], reduction=se_reduction),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 4: 9x9 -> 4x4
        self.layer4 = nn.Sequential(
            nn.Conv2d(widths[2], widths[3], kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(widths[3]),
            nn.LeakyReLU(slope, inplace=True),
            MADSELayer(widths[3], reduction=se_reduction),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Readout (Corrected Decoupled Isomorphic) ---
        # We use separate 1x1 convolutions for Stage 3 and Stage 4 features.
        self.proj3 = nn.Conv2d(widths[2], proj_dim, kernel_size=1, bias=True)
        self.proj4 = nn.Conv2d(widths[3], proj_dim, kernel_size=1, bias=True)

        # --- Classification Head ---
        # Input: Feature Vector (256) + Angle (1) = 257
        # Feature Vector construction:
        #   Stage 3: Max(64) + Min(64)
        #   Stage 4: Max(64) + Min(64)
        self.head = nn.Sequential(
            nn.Linear(feature_dim + 1, 256),
            nn.LeakyReLU(slope, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Kaiming Uniform initialization for LeakyReLU.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight,
                    a=Config.LEAKY_RELU_SLOPE,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight,
                    a=Config.LEAKY_RELU_SLOPE,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, angle):
        """
        Forward pass.
        Args:
            x: Image tensor (B, 3, 75, 75)
            angle: Incidence angle tensor (B,)
        """
        # Backbone Forward
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # --- Readout Stage 3 ---
        # Project
        p3 = self.proj3(x3)  # (B, 64, H3, W3)
        # Global Max Pooling
        max3 = F.adaptive_max_pool2d(p3, 1).view(p3.size(0), -1)
        # Global Min Pooling (Flatten spatial dims -> Min -> Values)
        min3 = p3.view(p3.size(0), p3.size(1), -1).min(dim=2)[0]

        # --- Readout Stage 4 ---
        # Project
        p4 = self.proj4(x4)  # (B, 64, H4, W4)
        # Global Max Pooling
        max4 = F.adaptive_max_pool2d(p4, 1).view(p4.size(0), -1)
        # Global Min Pooling
        min4 = p4.view(p4.size(0), p4.size(1), -1).min(dim=2)[0]

        # Concatenate Image Features
        # Size: 64+64+64+64 = 256
        features = torch.cat([max3, min3, max4, min4], dim=1)

        # --- Fusion ---
        # Concatenate with raw angle
        angle = angle.view(-1, 1)
        fused = torch.cat([features, angle], dim=1)

        # Classification
        out = self.head(fused)

        # Return flattened logits (B,)
        return out.view(-1)
