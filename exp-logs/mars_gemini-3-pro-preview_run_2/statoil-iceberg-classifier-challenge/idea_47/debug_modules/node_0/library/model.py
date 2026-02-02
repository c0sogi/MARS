import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.layers import CBAM, DualPooling


class WideBlock(nn.Module):
    """
    Wide Convolutional Block: 3x3 Conv -> BN -> ReLU.
    Maps input channels to 'width' filters.
    """

    def __init__(self, in_channels, out_channels):
        super(WideBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class PCWBN(nn.Module):
    """
    Pyramidal Context Wide-Body Network (PC-WBN).

    Features:
    - Wide-Body Backbone with Delayed Integration.
    - CBAM Attention (Mixed Pooling).
    - Dual-Stream Pooling (Max + Min).
    - Tri-Path Pyramidal Readout (Detail, Context, Global).
    - Metadata Embedding Branch.
    """

    def __init__(self):
        super(PCWBN, self).__init__()

        # --- Configuration ---
        filters = Config.BACKBONE_FILTERS  # 128
        detail_dim = Config.READOUT_DIM_DETAIL  # 32
        context_dim = Config.READOUT_DIM_CONTEXT  # 32
        meta_dim = Config.META_EMBED_DIM  # 32
        dropout = Config.DROPOUT_RATE  # 0.5

        # --- Visual Branch (Backbone) ---

        # Stage 1: 75x75 -> 37x37
        # Input: 3 channels (Band1, Band2, Mean)
        self.s1_conv = WideBlock(Config.IN_CHANNELS, filters)
        self.s1_cbam = CBAM(filters)
        self.s1_pool = DualPooling(
            kernel_size=2, stride=2
        )  # Output channels: filters * 2 = 256

        # Stage 2: 37x37 -> 18x18
        # Input: 256 channels -> Map to 128 (Delayed Integration)
        self.s2_conv = WideBlock(filters * 2, filters)
        self.s2_cbam = CBAM(filters)
        self.s2_pool = DualPooling(kernel_size=2, stride=2)  # Output: 256

        # Stage 3: 18x18 -> 9x9
        self.s3_conv = WideBlock(filters * 2, filters)
        self.s3_cbam = CBAM(filters)
        self.s3_pool = DualPooling(kernel_size=2, stride=2)  # Output: 256

        # Stage 4: 9x9 -> 4x4
        self.s4_conv = WideBlock(filters * 2, filters)
        self.s4_cbam = CBAM(filters)
        self.s4_pool = DualPooling(kernel_size=2, stride=2)  # Output: 256

        # Calculate dimensions
        self.final_spatial_dim = 4
        self.backbone_out_channels = filters * 2  # 256

        # --- Tri-Path Pyramidal Readout ---

        # Path A: Detail Stream (1x1 Conv -> Flatten)
        # Preserves pixel-wise signal fidelity
        self.path_a_conv = nn.Conv2d(
            self.backbone_out_channels, detail_dim, kernel_size=1, bias=False
        )
        self.path_a_flat_dim = (
            detail_dim * self.final_spatial_dim * self.final_spatial_dim
        )
        self.path_a_bn = nn.BatchNorm1d(self.path_a_flat_dim)

        # Path B: Context Stream (3x3 Conv -> Flatten)
        # Integrates spatial adjacency
        self.path_b_conv = nn.Conv2d(
            self.backbone_out_channels,
            context_dim,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.path_b_flat_dim = (
            context_dim * self.final_spatial_dim * self.final_spatial_dim
        )
        self.path_b_bn = nn.BatchNorm1d(self.path_b_flat_dim)

        # Path C: Global Stream (GAP)
        # Captures translation-invariant stats
        self.path_c_pool = nn.AdaptiveAvgPool2d(1)
        self.path_c_dim = self.backbone_out_channels
        self.path_c_bn = nn.BatchNorm1d(self.path_c_dim)

        # --- Metadata Branch ---
        self.meta_fc1 = nn.Linear(1, meta_dim)
        self.meta_fc2 = nn.Linear(meta_dim, meta_dim)
        self.meta_bn = nn.BatchNorm1d(meta_dim)
        self.meta_relu = nn.ReLU(inplace=True)

        # --- Fusion Head ---
        fusion_in_dim = (
            self.path_a_flat_dim + self.path_b_flat_dim + self.path_c_dim + meta_dim
        )

        self.head_fc1 = nn.Linear(fusion_in_dim, 512)
        self.head_bn = nn.BatchNorm1d(512)
        self.head_relu = nn.ReLU(inplace=True)
        self.head_dropout = nn.Dropout(dropout)
        self.head_out = nn.Linear(512, 1)  # Output logits

    def forward(self, x, inc_angle):
        """
        Forward pass of the PC-WBN.

        Args:
            x (torch.Tensor): Image tensor of shape (B, 3, 75, 75).
            inc_angle (torch.Tensor): Incidence angle tensor of shape (B,).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # --- Backbone ---
        # Stage 1
        x = self.s1_conv(x)
        x = self.s1_cbam(x)
        x = self.s1_pool(x)  # (B, 256, 37, 37)

        # Stage 2
        x = self.s2_conv(x)
        x = self.s2_cbam(x)
        x = self.s2_pool(x)  # (B, 256, 18, 18)

        # Stage 3
        x = self.s3_conv(x)
        x = self.s3_cbam(x)
        x = self.s3_pool(x)  # (B, 256, 9, 9)

        # Stage 4
        x = self.s4_conv(x)
        x = self.s4_cbam(x)
        x = self.s4_pool(x)  # (B, 256, 4, 4)

        # --- Readout ---
        # Path A (Detail)
        a = self.path_a_conv(x)
        a = a.view(a.size(0), -1)
        a = self.path_a_bn(a)

        # Path B (Context)
        b = self.path_b_conv(x)
        b = b.view(b.size(0), -1)
        b = self.path_b_bn(b)

        # Path C (Global)
        c = self.path_c_pool(x)
        c = c.view(c.size(0), -1)
        c = self.path_c_bn(c)

        # --- Metadata ---
        # Ensure inc_angle is (B, 1)
        m = inc_angle.view(-1, 1)
        m = F.relu(self.meta_fc1(m))
        m = self.meta_fc2(m)
        m = self.meta_bn(m)
        m = self.meta_relu(m)

        # --- Fusion ---
        f = torch.cat([a, b, c, m], dim=1)
        f = self.head_fc1(f)
        f = self.head_bn(f)
        f = self.head_relu(f)
        f = self.head_dropout(f)

        logits = self.head_out(f)
        return logits
