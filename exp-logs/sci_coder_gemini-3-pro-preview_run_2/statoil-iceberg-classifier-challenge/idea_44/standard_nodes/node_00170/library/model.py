import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Refines features using Channel and Spatial attention.
    Uses Mixed Pooling (Max + Avg) for stability.
    """

    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel Attention
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # --- Channel Attention ---
        b, c, _, _ = x.size()

        # Global Average Pooling
        avg_pool = F.avg_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
        ).view(b, c)
        avg_out = self.mlp(avg_pool)

        # Global Max Pooling
        max_pool = F.max_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
        ).view(b, c)
        max_out = self.mlp(max_pool)

        # Combine and Sigmoid
        channel_att = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att

        # --- Spatial Attention ---
        # Channel-wise Avg and Max
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)

        # Conv and Sigmoid
        spatial_att = torch.sigmoid(self.conv_spatial(spatial_in))
        x = x * spatial_att

        return x


class InputAnchoredWideBodyNet(nn.Module):
    """
    Input-Anchored Wide-Body Network (IA-WBN).
    Features:
    - Wide-Body Backbone (128 filters)
    - Dual-Stream Pooling (Max + Min)
    - Input Anchor Branch (Raw signal stats)
    - Normalized Dual-Path Readout
    """

    def __init__(self):
        super(InputAnchoredWideBodyNet, self).__init__()

        # --- Visual Branch Backbone ---
        # Stage 1: 3 -> 128 -> DualPool(256)
        self.conv1 = nn.Conv2d(3, config.VISUAL_FILTERS, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(config.VISUAL_FILTERS)
        self.cbam1 = CBAM(config.VISUAL_FILTERS)

        # Stage 2: 256 -> 128 -> DualPool(256)
        self.conv2 = nn.Conv2d(
            config.VISUAL_FILTERS * 2, config.VISUAL_FILTERS, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(config.VISUAL_FILTERS)
        self.cbam2 = CBAM(config.VISUAL_FILTERS)

        # Stage 3: 256 -> 128 -> DualPool(256)
        self.conv3 = nn.Conv2d(
            config.VISUAL_FILTERS * 2, config.VISUAL_FILTERS, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(config.VISUAL_FILTERS)
        self.cbam3 = CBAM(config.VISUAL_FILTERS)

        # Stage 4: 256 -> 128 -> DualPool(256)
        self.conv4 = nn.Conv2d(
            config.VISUAL_FILTERS * 2, config.VISUAL_FILTERS, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(config.VISUAL_FILTERS)
        self.cbam4 = CBAM(config.VISUAL_FILTERS)

        # --- Readout Paths ---
        # Path A: Spatial Context (Conv -> Flatten -> BN)
        # Input: 256 channels, Output: 64 channels
        self.path_a_conv = nn.Conv2d(
            config.VISUAL_FILTERS * 2, config.PATH_A_UNITS, kernel_size=3, padding=1
        )
        # Assuming input 75x75 -> 37 -> 18 -> 9 -> 4. Output spatial dim is 4x4.
        self.path_a_bn = nn.BatchNorm1d(config.PATH_A_UNITS * 4 * 4)

        # Path B: Robust Intensity (GAP -> BN)
        self.path_b_bn = nn.BatchNorm1d(config.VISUAL_FILTERS * 2)

        # --- Metadata Branch ---
        # Linear -> ReLU -> Linear -> BN -> ReLU
        self.meta_fc1 = nn.Linear(1, 16)
        self.meta_fc2 = nn.Linear(16, 32)
        self.meta_bn = nn.BatchNorm1d(32)

        # --- Input Anchor Branch ---
        # 3 channels * (Max + Mean) = 6 inputs
        self.anchor_fc = nn.Linear(6, config.ANCHOR_HIDDEN_UNITS)
        self.anchor_bn = nn.BatchNorm1d(config.ANCHOR_HIDDEN_UNITS)

        # --- Fusion Head ---
        # Calculate fusion dimension
        # Path A: 64 * 4 * 4 = 1024
        # Path B: 256
        # Meta: 32
        # Anchor: 16
        fusion_dim = (
            (config.PATH_A_UNITS * 4 * 4)
            + (config.VISUAL_FILTERS * 2)
            + 32
            + config.ANCHOR_HIDDEN_UNITS
        )

        self.fusion_fc = nn.Linear(fusion_dim, 512)
        self.fusion_bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x, inc_angle):
        """
        Args:
            x: (N, 3, 75, 75) image tensor
            inc_angle: (N,) incidence angle tensor
        """
        # --- Input Anchor Branch ---
        # Compute global stats on raw input x (N, 3, 75, 75)
        x_flat = x.view(x.size(0), 3, -1)

        # Max per channel (N, 3)
        max_vals = x_flat.max(dim=2)[0]
        # Mean per channel (N, 3)
        mean_vals = x_flat.mean(dim=2)

        anchor_in = torch.cat([max_vals, mean_vals], dim=1)  # (N, 6)
        anchor_out = F.relu(self.anchor_bn(self.anchor_fc(anchor_in)))  # (N, 16)

        # --- Visual Branch ---
        # Stage 1
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.cbam1(out)
        # Dual Pooling (Max + Min)
        p_max = F.max_pool2d(out, 2)
        p_min = -F.max_pool2d(-out, 2)
        out = torch.cat([p_max, p_min], dim=1)  # (N, 256, 37, 37)

        # Stage 2
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.cbam2(out)
        p_max = F.max_pool2d(out, 2)
        p_min = -F.max_pool2d(-out, 2)
        out = torch.cat([p_max, p_min], dim=1)  # (N, 256, 18, 18)

        # Stage 3
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.cbam3(out)
        p_max = F.max_pool2d(out, 2)
        p_min = -F.max_pool2d(-out, 2)
        out = torch.cat([p_max, p_min], dim=1)  # (N, 256, 9, 9)

        # Stage 4
        out = F.relu(self.bn4(self.conv4(out)))
        out = self.cbam4(out)
        p_max = F.max_pool2d(out, 2)
        p_min = -F.max_pool2d(-out, 2)
        out = torch.cat([p_max, p_min], dim=1)  # (N, 256, 4, 4)

        # Path A (Spatial)
        path_a = self.path_a_conv(out)  # (N, 64, 4, 4)
        path_a = path_a.view(path_a.size(0), -1)  # Flatten
        path_a = self.path_a_bn(path_a)

        # Path B (Intensity)
        path_b = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)  # (N, 256)
        path_b = self.path_b_bn(path_b)

        # --- Metadata Branch ---
        inc = inc_angle.view(-1, 1)
        meta = F.relu(self.meta_fc1(inc))
        meta = F.relu(self.meta_bn(self.meta_fc2(meta)))  # (N, 32)

        # --- Fusion ---
        fused = torch.cat([path_a, path_b, meta, anchor_out], dim=1)
        fused = F.relu(self.fusion_bn(self.fusion_fc(fused)))
        fused = self.dropout(fused)

        logits = self.classifier(fused)

        return logits
