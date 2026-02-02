import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Mixed Pooling (Max + Avg) across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class DualPooling(nn.Module):
    def __init__(self):
        super(DualPooling, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        # Max Pooling (Peaks)
        max_p = self.pool(x)
        # Min Pooling (Shadows) implemented as -MaxPool(-x)
        min_p = -self.pool(-x)
        # Concatenate: expands channel depth 2x
        return torch.cat([max_p, min_p], dim=1)


class WideBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WideBlock, self).__init__()
        # Wide Convolution: Maps input to 128 filters
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Pre-Pooling Attention
        self.cbam = CBAM(out_channels)

        # Dual-Stream Pooling (expands out_channels -> 2*out_channels)
        self.dual_pool = DualPooling()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)
        x = self.dual_pool(x)
        return x


class RDP_WBN(nn.Module):
    def __init__(self):
        super(RDP_WBN, self).__init__()

        # --- Visual Branch (Wide-Body Delayed-Integration Backbone) ---
        # Block 1: Input 3 -> 128 (DualPool -> 256)
        self.block1 = WideBlock(config.NUM_CHANNELS, config.BACKBONE_FILTERS)

        # Block 2: Input 256 -> 128 (DualPool -> 256)
        self.block2 = WideBlock(config.BACKBONE_FILTERS * 2, config.BACKBONE_FILTERS)

        # Block 3: Input 256 -> 128 (DualPool -> 256)
        self.block3 = WideBlock(config.BACKBONE_FILTERS * 2, config.BACKBONE_FILTERS)

        # Block 4: Input 256 -> 128 (DualPool -> 256)
        self.block4 = WideBlock(config.BACKBONE_FILTERS * 2, config.BACKBONE_FILTERS)

        # --- Robust Dual-Path Readout ---
        # Path A (Spatial Context): Conv 3x3 to compress 256 -> 48
        self.readout_conv = nn.Conv2d(
            config.BACKBONE_FILTERS * 2,
            config.READOUT_PATH_A_FILTERS,
            kernel_size=3,
            padding=1,
        )

        # Path B (Robust Intensity): Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # --- Metadata Branch ---
        # MLP for inc_angle
        self.meta_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
        )

        # --- Fusion Head ---
        # Visual dim: Path A (48 * 4 * 4 = 768) + Path B (256) = 1024
        # Note: 75x75 input -> pool 4 times -> 4x4 spatial dim
        self.visual_dim = (config.READOUT_PATH_A_FILTERS * 4 * 4) + (
            config.BACKBONE_FILTERS * 2
        )
        self.meta_dim = 32

        self.fusion_head = nn.Sequential(
            nn.Linear(self.visual_dim + self.meta_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(512, config.NUM_CLASSES),
        )

    def forward(self, x, inc_angle):
        # --- Visual Branch ---
        x = self.block1(x)  # 75 -> 37
        x = self.block2(x)  # 37 -> 18
        x = self.block3(x)  # 18 -> 9
        x = self.block4(x)  # 9 -> 4

        # --- Dual-Path Readout ---
        # Path A: Spatial Context
        # x is (B, 256, 4, 4)
        path_a = self.readout_conv(x)  # (B, 48, 4, 4)
        path_a = path_a.view(path_a.size(0), -1)  # Flatten -> (B, 768)

        # Path B: Robust Intensity
        path_b = self.global_avg_pool(x)  # (B, 256, 1, 1)
        path_b = path_b.view(path_b.size(0), -1)  # Flatten -> (B, 256)

        # Fuse Visual Paths
        visual_feat = torch.cat([path_a, path_b], dim=1)  # (B, 1024)

        # --- Metadata Branch ---
        # inc_angle is (B,) or (B, 1)
        if inc_angle.dim() == 1:
            inc_angle = inc_angle.unsqueeze(1)
        meta_feat = self.meta_mlp(inc_angle)

        # --- Final Fusion ---
        combined = torch.cat([visual_feat, meta_feat], dim=1)
        out = self.fusion_head(combined)

        return out
