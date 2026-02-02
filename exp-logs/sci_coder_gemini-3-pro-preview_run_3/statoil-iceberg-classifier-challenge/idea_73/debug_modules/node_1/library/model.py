import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.utils import calculate_mad


class HybridSE(nn.Module):
    """
    Hybrid Squeeze-and-Excitation block.
    Uses Global Average Pooling to squeeze spatial information, followed by
    a two-layer MLP to recalibrate channel-wise feature responses.
    """

    def __init__(self, channels, reduction=16):
        super(HybridSE, self).__init__()
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


class ADSICNN(nn.Module):
    """
    Asymmetric Dual-Statistic Isomorphic CNN (ADSI-CNN).

    Features a 4-stage Plain CNN backbone with asymmetric readouts:
    - Stage 3 (High Res): Extracts Peak (Max) and Shadow (Min) statistics.
    - Stage 4 (Global Field): Extracts Peak (Max) and Texture (MAD) statistics.

    This architecture is designed to capture physical properties of icebergs
    (shadow casting, surface roughness) while maintaining parameter efficiency.
    """

    def __init__(self, base_width=None, dropout_rate=None):
        super(ADSICNN, self).__init__()

        # Use Config defaults if not provided
        self.base_width = base_width if base_width is not None else Config.BASE_WIDTH
        self.dropout_rate = (
            dropout_rate if dropout_rate is not None else Config.DROPOUT_RATE
        )

        def make_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.1, inplace=True),
                HybridSE(out_c),
                nn.MaxPool2d(2),
            )

        # Stage 1: 75x75 -> 37x37
        self.stage1 = make_block(Config.CHANNELS, self.base_width)

        # Stage 2: 37x37 -> 18x18
        self.stage2 = make_block(self.base_width, 128)

        # Stage 3: 18x18 -> 9x9 (High resolution for Shadows)
        # Note: We separate conv and pool in the reference, but here we can keep them
        # combined as long as we use the output of the block (which is pooled).
        # The description says "Stage 3 (where spatial resolution is higher)".
        # 9x9 is higher than 4x4.
        self.stage3 = make_block(128, 128)

        # Stage 4: 9x9 -> 4x4 (Global receptive field for Texture)
        self.stage4 = make_block(128, 128)

        # Decoupled Projections (1x1 Convs)
        # Map 128 channels to 64 for specific statistic extraction
        self.proj_s3 = nn.Conv2d(128, 64, kernel_size=1)
        self.proj_s4 = nn.Conv2d(128, 64, kernel_size=1)

        # Classification Head
        # Input: S3_Max(64) + S3_Min(64) + S4_Max(64) + S4_MAD(64) + Angle(1) = 257
        self.head = nn.Sequential(
            nn.Linear(257, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 1),
        )

    def forward(self, x, angle):
        # Backbone Forward Pass
        x = self.stage1(x)
        x = self.stage2(x)

        s3_out = self.stage3(x)  # Output is 9x9
        s4_out = self.stage4(s3_out)  # Output is 4x4

        # --- Asymmetric Readout ---

        # Stage 3: Shape & Shadow Focus (9x9)
        p3 = self.proj_s3(s3_out)

        # Global Max Pooling (Peaks)
        s3_max = F.adaptive_max_pool2d(p3, 1).view(p3.size(0), -1)

        # Global Min Pooling (Shadows)
        # Implemented as -Max(-x)
        s3_min = -F.adaptive_max_pool2d(-p3, 1).view(p3.size(0), -1)

        # Stage 4: Existence & Texture Focus (4x4)
        p4 = self.proj_s4(s4_out)

        # Global Max Pooling (Peaks)
        s4_max = F.adaptive_max_pool2d(p4, 1).view(p4.size(0), -1)

        # Global MAD Pooling (Texture via Mean Absolute Deviation)
        # Calculate MAD over spatial dimensions (2, 3)
        s4_mad = calculate_mad(p4, dim=(2, 3), keepdim=False)
        # Ensure it is flattened (B, 64)
        s4_mad = s4_mad.view(p4.size(0), -1)

        # --- Feature Fusion ---

        # Reshape angle for concatenation
        angle = angle.view(-1, 1)

        # Concatenate all features
        features = torch.cat([s3_max, s3_min, s4_max, s4_mad, angle], dim=1)

        # Classification
        return self.head(features)
