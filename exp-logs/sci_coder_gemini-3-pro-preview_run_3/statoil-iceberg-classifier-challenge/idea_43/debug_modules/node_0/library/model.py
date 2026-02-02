import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSE(nn.Module):
    """
    Hybrid Squeeze-and-Excitation Module.
    Uses Global Average Pooling for squeezing to be robust to speckle noise.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super(HybridSE, self).__init__()
        reduced_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for CAFP-CNN.
    Structure: Conv2d (bias=True) -> BN -> LeakyReLU -> HybridSE -> MaxPool.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        # Bias is explicitly retained to preserve initialization dynamics (Lesson 76)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        # LeakyReLU with negative slope 0.1 to preserve shadow information (Lesson 91)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.se = HybridSE(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class CAFPCNN(nn.Module):
    """
    Contrast-Aware Feature-Pyramid CNN (CAFP-CNN).

    Features:
    - 4-Stage Plain CNN Backbone.
    - Top-Down Feature Pyramid Fusion (Stage 4 + Stage 3).
    - Dual-Polarity Pooling (Max + Min) to capture Peak-to-Shadow contrast.
    - Raw Incidence Angle injection in the classifier head.
    """

    def __init__(self):
        super(CAFPCNN, self).__init__()

        # --- Backbone ---
        # Input: (B, 3, 75, 75)
        # Stage 1: 64 channels, 75 -> 37
        self.stage1 = ConvBlock(3, 64)
        # Stage 2: 128 channels, 37 -> 18
        self.stage2 = ConvBlock(64, 128)
        # Stage 3: 128 channels, 18 -> 9
        self.stage3 = ConvBlock(128, 128)
        # Stage 4: 128 channels, 9 -> 4
        self.stage4 = ConvBlock(128, 128)

        # --- Feature Pyramid Fusion ---
        # Fuses Stage 4 (Abstract) with Stage 3 (Spatial)
        # Input: 128 (Stage 3) + 128 (Stage 4) = 256
        # Output: 128 channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # --- Classification Head ---
        # Input: 128 (Max Pool) + 128 (Min Pool) + 1 (Angle) = 257
        self.head = nn.Sequential(
            nn.Linear(257, 256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, inc_angle: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (B, 3, 75, 75)
            inc_angle: Incidence angle tensor of shape (B,) or (B, 1)
        """
        # Backbone Forward Pass
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)  # Save for fusion (B, 128, 9, 9)
        s4 = self.stage4(s3)  # Final stage (B, 128, 4, 4)

        # Feature Pyramid Fusion
        # Upsample Stage 4 to match Stage 3 spatial dimensions
        s4_up = F.interpolate(s4, size=s3.shape[2:], mode="nearest")

        # Concatenate along channel dimension
        fused_map = torch.cat([s3, s4_up], dim=1)  # (B, 256, 9, 9)

        # Compress features
        fused_map = self.fusion_conv(fused_map)  # (B, 128, 9, 9)

        # Dual-Polarity Readout
        # 1. Global Max Pooling (Captures high backscatter peaks)
        max_pool = F.adaptive_max_pool2d(fused_map, 1).view(fused_map.size(0), -1)

        # 2. Global Min Pooling (Captures shadows/voids)
        # Implemented as -Max(-x)
        min_pool = -F.adaptive_max_pool2d(-fused_map, 1).view(fused_map.size(0), -1)

        # Concatenate pooled features (128 + 128 = 256)
        features = torch.cat([max_pool, min_pool], dim=1)

        # Prepare Incidence Angle
        if inc_angle.dim() == 1:
            inc_angle = inc_angle.view(-1, 1)

        # Concatenate Angle (256 + 1 = 257)
        # Note: Angle is not normalized, as per design
        final_input = torch.cat([features, inc_angle], dim=1)

        # Classification
        logits = self.head(final_input)

        return logits
