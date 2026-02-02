import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


# ==========================================
# ATTENTION MODULES (CBAM)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Ensure hidden planes is at least 1
        hidden_planes = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)

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
        # Mixed pooling for spatial attention: Avg and Max
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ==========================================
# POOLING MODULES
# ==========================================
class DualPooling(nn.Module):
    """
    Applies Max Pooling and Min Pooling (via negated Max Pooling) and concatenates along channel dimension.
    Doubles the channel count.
    """

    def __init__(self, kernel_size=2, stride=2):
        super(DualPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # Max Pooling
        out_max = self.max_pool(x)
        # Min Pooling: -max_pool(-x)
        out_min = -self.max_pool(-x)
        return torch.cat([out_max, out_min], dim=1)


class TopKAvgPooling(nn.Module):
    """
    Sorts pixels in spatial dimensions and computes the average of the top K pixels.
    Acts as a robust 'Soft Max Pooling'.
    """

    def __init__(self, k=4):
        super(TopKAvgPooling, self).__init__()
        self.k = k

    def forward(self, x):
        # x: (N, C, H, W)
        n, c, h, w = x.size()
        # Flatten spatial dims: (N, C, H*W)
        x_flat = x.view(n, c, -1)

        # Handle case where H*W < k (safety check)
        k_actual = min(self.k, x_flat.size(2))

        # Get top k values along the spatial dimension
        top_k_vals, _ = torch.topk(x_flat, k_actual, dim=2)

        # Compute mean of these top K pixels
        out = torch.mean(top_k_vals, dim=2)  # (N, C)
        return out


# ==========================================
# BUILDING BLOCKS
# ==========================================
class WideBodyBlock(nn.Module):
    """
    Implements the Delayed-Integration Block Topology:
    Wide Conv -> BN -> ReLU -> CBAM -> DualPooling
    """

    def __init__(self, in_channels, out_channels):
        super(WideBodyBlock, self).__init__()

        # Wide Convolution: Maps input to 'out_channels' (typically 128)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Pre-Pooling Attention
        self.cbam = CBAM(out_channels)

        # Dual Pooling (Expands channels by 2x)
        self.dual_pool = DualPooling(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.cbam(x)

        x = self.dual_pool(x)
        return x


class MetadataBranch(nn.Module):
    """
    Deep Normalized Embedding for Incidence Angle.
    """

    def __init__(self, output_dim=32):
        super(MetadataBranch, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# MAIN MODEL
# ==========================================
class TKA_WBN(nn.Module):
    """
    Top-K Augmented Wide-Body Network.
    """

    def __init__(self):
        super(TKA_WBN, self).__init__()

        # --- Visual Branch (Wide-Body Delayed-Integration Backbone) ---
        # Block 1: Input 3 -> Conv 64 -> DualPool -> 128
        # Output Size: 75 -> 37
        self.block1 = WideBodyBlock(in_channels=config.NUM_CHANNELS, out_channels=64)

        # Block 2: Input 128 -> Conv 128 -> DualPool -> 256
        # Output Size: 37 -> 18
        self.block2 = WideBodyBlock(in_channels=128, out_channels=config.NUM_FILTERS)

        # Block 3: Input 256 -> Conv 128 -> DualPool -> 256
        # Output Size: 18 -> 9
        self.block3 = WideBodyBlock(in_channels=256, out_channels=config.NUM_FILTERS)

        # Block 4: Input 256 -> Conv 128 -> DualPool -> 256
        # Output Size: 9 -> 4
        self.block4 = WideBodyBlock(in_channels=256, out_channels=config.NUM_FILTERS)

        # Final Feature Map Size: (N, 256, 4, 4)

        # --- Tri-Path Readout ---

        # Path A: Spatial Context
        # Conv 256 -> 64 (3x3, pad=1) to preserve spatial grid, then flatten
        self.path_a_conv = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        # Flatten size: 64 * 4 * 4 = 1024
        self.path_a_bn = nn.BatchNorm1d(1024)

        # Path B: Background/Global Intensity (Global Average Pooling)
        self.path_b_pool = nn.AdaptiveAvgPool2d(1)
        self.path_b_bn = nn.BatchNorm1d(256)

        # Path C: Robust Target Intensity (Top-K Average Pooling)
        self.path_c_pool = TopKAvgPooling(k=config.TOP_K)
        self.path_c_bn = nn.BatchNorm1d(256)

        # --- Metadata Branch ---
        self.meta_branch = MetadataBranch(output_dim=32)

        # --- Fusion Head ---
        # Input dims: Path A (1024) + Path B (256) + Path C (256) + Meta (32) = 1568
        fusion_dim = 1024 + 256 + 256 + 32

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(512, 1),
        )

    def forward(self, x_img, x_inc):
        # Visual Backbone
        x = self.block1(x_img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x shape: (N, 256, 4, 4)

        # Path A: Spatial Context
        out_a = self.path_a_conv(x)  # (N, 64, 4, 4)
        out_a = out_a.view(out_a.size(0), -1)  # Flatten -> (N, 1024)
        out_a = self.path_a_bn(out_a)

        # Path B: Global Average
        out_b = self.path_b_pool(x)  # (N, 256, 1, 1)
        out_b = out_b.view(out_b.size(0), -1)  # (N, 256)
        out_b = self.path_b_bn(out_b)

        # Path C: Top-K Average
        out_c = self.path_c_pool(x)  # (N, 256)
        out_c = self.path_c_bn(out_c)

        # Metadata
        out_meta = self.meta_branch(x_inc)  # (N, 32)

        # Fusion
        fused = torch.cat([out_a, out_b, out_c, out_meta], dim=1)

        logits = self.fusion_head(fused)
        return torch.sigmoid(logits)
