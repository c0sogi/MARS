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

        # --- Dual-Path Readout ---
        # Cite solution_lesson_node_00179: Removed Detail Stream (1x1 Conv) to avoid overfitting to speckle noise.
        # Cite solution_lesson_node_00123: Using Dual-Path (Context + Global) for robustness.

        # Path 1: Context Stream (3x3 Conv -> Flatten)
        # Integrates spatial adjacency
        self.path_context_conv = nn.Conv2d(
            self.backbone_out_channels,
            context_dim,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.path_context_flat_dim = (
            context_dim * self.final_spatial_dim * self.final_spatial_dim
        )
        self.path_context_bn = nn.BatchNorm1d(self.path_context_flat_dim)

        # Path 2: Global Stream (GAP)
        # Captures translation-invariant stats
        self.path_global_pool = nn.AdaptiveAvgPool2d(1)
        self.path_global_dim = self.backbone_out_channels
        self.path_global_bn = nn.BatchNorm1d(self.path_global_dim)

        # --- Metadata Branch ---
        self.meta_fc1 = nn.Linear(1, meta_dim)
        self.meta_fc2 = nn.Linear(meta_dim, meta_dim)
        self.meta_bn = nn.BatchNorm1d(meta_dim)
        self.meta_relu = nn.ReLU(inplace=True)

        # --- Fusion Head ---
        fusion_in_dim = self.path_context_flat_dim + self.path_global_dim + meta_dim

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
        # Path 1: Context
        ctx = self.path_context_conv(x)
        ctx = ctx.view(ctx.size(0), -1)
        ctx = self.path_context_bn(ctx)

        # Path 2: Global
        glb = self.path_global_pool(x)
        glb = glb.view(glb.size(0), -1)
        glb = self.path_global_bn(glb)

        # --- Metadata ---
        # Ensure inc_angle is (B, 1)
        m = inc_angle.view(-1, 1)
        m = F.relu(self.meta_fc1(m))
        m = self.meta_fc2(m)
        m = self.meta_bn(m)
        m = self.meta_relu(m)

        # --- Fusion ---
        f = torch.cat([ctx, glb, m], dim=1)
        f = self.head_fc1(f)
        f = self.head_bn(f)
        f = self.head_relu(f)
        f = self.head_dropout(f)

        logits = self.head_out(f)
        return logits
