import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DualStreamPooling(nn.Module):
    """
    Dual-Stream Pooling: Applies Max Pooling (Peaks) and Min Pooling (Shadows)
    and concatenates the outputs. Expands channel depth by 2x.
    """

    def __init__(self):
        super(DualStreamPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        # Max Pooling
        max_p = self.max_pool(x)
        # Min Pooling implemented as -MaxPool(-x)
        min_p = -self.max_pool(-x)
        return torch.cat([max_p, min_p], dim=1)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module with Mixed Pooling (Max + Avg).
    Applied before pooling to refine features at high resolution.
    """

    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()

        # Channel Attention
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        # --- Channel Attention ---
        # Global Avg Pool
        avg_pool = F.avg_pool2d(x, (h, w)).view(b, c)
        # Global Max Pool
        max_pool = F.max_pool2d(x, (h, w)).view(b, c)

        # Shared MLP
        channel_att = self.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(
            b, c, 1, 1
        )
        x = x * channel_att

        # --- Spatial Attention ---
        # Channel Pool (Avg + Max)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)

        spatial_att = self.sigmoid(self.conv_spatial(spatial_in))
        x = x * spatial_att

        return x


class WideBlock(nn.Module):
    """
    Wide-Body Delayed-Integration Block.
    Structure: Conv(Wide) -> BN -> ReLU -> CBAM -> DualPooling.
    """

    def __init__(self, in_channels, out_channels):
        super(WideBlock, self).__init__()
        # Wide Convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

        # Pre-Pooling Attention
        self.cbam = CBAM(out_channels)

        # Dual-Stream Pooling (Expands channels 2x)
        self.pool = DualStreamPooling()

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.cbam(x)
        x = self.pool(x)
        return x


class MDSWBN(nn.Module):
    """
    Multi-Depth Statistical Wide-Body Network.
    Features a Split-Branch Topology with a Multi-Depth Statistical Readout.
    """

    def __init__(self):
        super(MDSWBN, self).__init__()

        # --- Visual Branch (Backbone) ---
        # Block 1: Input 3 -> 128 (Output 256 due to DualPooling)
        self.block1 = WideBlock(3, 128)

        # Block 2: Input 256 -> 128 (Output 256) - Delayed Integration
        self.block2 = WideBlock(256, 128)

        # Block 3: Input 256 -> 128 (Output 256)
        self.block3 = WideBlock(256, 128)

        # Block 4: Input 256 -> 128 (Output 256)
        self.block4 = WideBlock(256, 128)

        # --- Readout Mechanisms ---
        # Path A (Spatial Context): Conv on Block 4
        self.readout_conv = nn.Conv2d(256, 48, kernel_size=3, padding=1)

        # Path B (Multi-Depth Invariance): GAP on Blocks 2, 3, 4
        # No learnable layers needed for GAP, logic in forward

        # --- Metadata Branch ---
        self.meta_mlp = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()
        )

        # --- Fusion Head ---
        # Visual Dimension Calculation:
        # Path A: 4x4 spatial * 48 channels = 768
        # Path B: 256 (B2) + 256 (B3) + 256 (B4) = 768
        # Total Visual: 1536
        # Meta: 32
        # Total Input: 1568
        self.head = nn.Sequential(
            nn.Linear(1568, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(128, 1),
        )

    def forward(self, x, inc_angle):
        # Backbone Forward Pass
        x1_out = self.block1(x)  # 37x37, 256ch
        x2_out = self.block2(x1_out)  # 18x18, 256ch
        x3_out = self.block3(x2_out)  # 9x9, 256ch
        x4_out = self.block4(x3_out)  # 4x4, 256ch

        # --- Multi-Depth Statistical Readout ---

        # Path A: Spatial Context (from Block 4)
        path_a = self.readout_conv(x4_out)  # 4x4x48
        path_a = path_a.view(path_a.size(0), -1)  # Flatten -> 768

        # Path B: Multi-Depth Invariance (GAP from B2, B3, B4)
        gap2 = F.adaptive_avg_pool2d(x2_out, (1, 1)).view(x2_out.size(0), -1)  # 256
        gap3 = F.adaptive_avg_pool2d(x3_out, (1, 1)).view(x3_out.size(0), -1)  # 256
        gap4 = F.adaptive_avg_pool2d(x4_out, (1, 1)).view(x4_out.size(0), -1)  # 256
        path_b = torch.cat([gap2, gap3, gap4], dim=1)  # 768

        # Visual Fusion
        visual_feat = torch.cat([path_a, path_b], dim=1)  # 1536

        # --- Metadata Processing ---
        meta_feat = self.meta_mlp(inc_angle)

        # --- Final Fusion & Classification ---
        combined = torch.cat([visual_feat, meta_feat], dim=1)
        out = self.head(combined)

        return torch.sigmoid(out)
