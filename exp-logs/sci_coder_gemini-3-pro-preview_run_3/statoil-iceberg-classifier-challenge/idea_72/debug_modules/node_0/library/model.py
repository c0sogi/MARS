import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalMADPool(nn.Module):
    """
    Global Mean Absolute Deviation (MAD) Pooling.
    Calculates Mean(|x - Mean(x)|) over spatial dimensions.
    Captures texture/roughness robustly.
    """

    def __init__(self):
        super(GlobalMADPool, self).__init__()

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Calculate spatial mean: (B, C, 1, 1)
        mean = x.mean(dim=[2, 3], keepdim=True)
        # Calculate MAD: (B, C)
        mad = torch.abs(x - mean).mean(dim=[2, 3])
        return mad


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, channels // reduction)
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
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for the Plain CNN Backbone.
    Conv2d (bias=True) -> BN -> LeakyReLU -> SE -> MaxPool
    """

    def __init__(self, in_channels, out_channels, reduction=16):
        super(ConvBlock, self).__init__()
        # Bias retained as per architectural design
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        # LeakyReLU with negative slope 0.1 to preserve shadow information
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.se = SEBlock(out_channels, reduction=reduction)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class TSICNN(nn.Module):
    """
    Tri-Statistic Isomorphic CNN (TSI-CNN).

    Features:
    - 4-Stage Plain CNN Backbone.
    - Selective Hierarchical Readout (Stage 3 & 4).
    - Decoupled Projections (128 -> 42 channels).
    - Tri-Statistic Pooling (Max, Min, MAD).
    - Raw Angle Fusion.
    """

    def __init__(self):
        super(TSICNN, self).__init__()

        # --- Backbone ---
        # Input: 3 channels (HH, HV, Avg)
        # Stage 1: 3 -> 64
        self.stage1 = ConvBlock(3, 64)
        # Stage 2: 64 -> 128
        self.stage2 = ConvBlock(64, 128)
        # Stage 3: 128 -> 128
        self.stage3 = ConvBlock(128, 128)
        # Stage 4: 128 -> 128
        self.stage4 = ConvBlock(128, 128)

        # --- Projections ---
        # Project 128 channels down to 42 to manage parameter budget
        self.proj3 = nn.Conv2d(128, 42, kernel_size=1, bias=True)
        self.proj4 = nn.Conv2d(128, 42, kernel_size=1, bias=True)

        # --- Pooling ---
        self.mad_pool = GlobalMADPool()
        # Max and Min pooling are performed functionally

        # --- Classification Head ---
        # Input Size Calculation:
        # 2 Stages (3 & 4) * 3 Stats (Max, Min, MAD) * 42 Channels = 252 features
        # + 1 Angle scalar = 253 input features
        self.head = nn.Sequential(
            nn.Linear(253, 256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Kaiming Uniform Initialization (Fan-In).
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
        # --- Backbone Forward ---
        x = self.stage1(x)
        x = self.stage2(x)

        # Extract Stage 3 Features
        x3 = self.stage3(x)

        # Extract Stage 4 Features
        x4 = self.stage4(x3)

        # --- Projections ---
        p3 = self.proj3(x3)  # Shape: (B, 42, H3, W3)
        p4 = self.proj4(x4)  # Shape: (B, 42, H4, W4)

        # --- Tri-Statistic Pooling (Stage 3) ---
        # 1. Global Max Pooling
        s3_max = F.adaptive_max_pool2d(p3, 1).view(p3.size(0), -1)
        # 2. Global Min Pooling (via negative max of negative)
        s3_min = -F.adaptive_max_pool2d(-p3, 1).view(p3.size(0), -1)
        # 3. Global MAD Pooling
        s3_mad = self.mad_pool(p3)

        # --- Tri-Statistic Pooling (Stage 4) ---
        s4_max = F.adaptive_max_pool2d(p4, 1).view(p4.size(0), -1)
        s4_min = -F.adaptive_max_pool2d(-p4, 1).view(p4.size(0), -1)
        s4_mad = self.mad_pool(p4)

        # --- Feature Concatenation ---
        # Concatenate all image-derived statistics
        img_features = torch.cat(
            [s3_max, s3_min, s3_mad, s4_max, s4_min, s4_mad], dim=1
        )

        # --- Angle Fusion ---
        # Ensure angle is shaped (B, 1)
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        # Concatenate image features with raw angle
        combined = torch.cat([img_features, angle], dim=1)

        # --- Classification ---
        logits = self.head(combined)

        # Squeeze to match target shape (B) if necessary, or return (B, 1)
        # Usually squeeze(1) is safer for BCEWithLogitsLoss with flat targets
        return logits.squeeze(1)
