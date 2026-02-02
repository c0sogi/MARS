import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.model_layers import WideBlock


class DMWBN(nn.Module):
    """
    Decoupled Morphological Wide-Body Network (DM-WBN).

    Features:
    - Wide-Body Backbone with Sustained Width (128 filters).
    - Dual-Stream Pooling (Max+Min) preserving Peaks and Shadows.
    - Decoupled Morphological Readout: Separately processes Peak and Shadow
      geometries before fusion.
    - Metadata integration for incidence angle.
    """

    def __init__(self):
        super(DMWBN, self).__init__()

        # ==========================================
        # 1. Wide-Body Delayed-Integration Backbone
        # ==========================================
        # Stage 1: Input (3) -> Conv(128) -> DualPool(256)
        # Output Spatial: 75x75 -> 37x37
        self.stage1 = WideBlock(
            in_channels=Config.CHANNELS, out_channels=Config.BACKBONE_FILTERS
        )

        # Stage 2: Input (256) -> Conv(128) -> DualPool(256)
        # Output Spatial: 37x37 -> 18x18
        self.stage2 = WideBlock(
            in_channels=Config.BACKBONE_FILTERS * 2,
            out_channels=Config.BACKBONE_FILTERS,
        )

        # Stage 3: Input (256) -> Conv(128) -> DualPool(256)
        # Output Spatial: 18x18 -> 9x9
        self.stage3 = WideBlock(
            in_channels=Config.BACKBONE_FILTERS * 2,
            out_channels=Config.BACKBONE_FILTERS,
        )

        # Stage 4: Input (256) -> Conv(128) -> DualPool(256)
        # Output Spatial: 9x9 -> 4x4
        self.stage4 = WideBlock(
            in_channels=Config.BACKBONE_FILTERS * 2,
            out_channels=Config.BACKBONE_FILTERS,
        )

        # ==========================================
        # 2. Decoupled Morphological Readout
        # ==========================================
        # Path A: Peak Morphology (First 128 channels)
        # Learns spatial filters specific to bright object shapes
        self.peak_conv = nn.Conv2d(
            in_channels=Config.BACKBONE_FILTERS,
            out_channels=Config.READOUT_DIM,
            kernel_size=3,
            padding=1,
        )

        # Path B: Shadow Morphology (Last 128 channels)
        # Learns spatial filters specific to signal voids/shadows
        self.shadow_conv = nn.Conv2d(
            in_channels=Config.BACKBONE_FILTERS,
            out_channels=Config.READOUT_DIM,
            kernel_size=3,
            padding=1,
        )

        # Path C: Global Intensity
        # Captures translation-invariant signal statistics
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ==========================================
        # 3. Metadata Branch
        # ==========================================
        # Dedicated MLP for incidence angle
        self.meta_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # ==========================================
        # 4. Fusion Head
        # ==========================================
        # Calculate Dimensions:
        # Spatial size at Stage 4 is 4x4
        spatial_dim = 4 * 4

        # Path A Flat Dim: 32 * 16 = 512
        self.flat_peak_dim = Config.READOUT_DIM * spatial_dim

        # Path B Flat Dim: 32 * 16 = 512
        self.flat_shadow_dim = Config.READOUT_DIM * spatial_dim

        # Path C Flat Dim: 256 (Global Avg of 256 channels)
        self.flat_global_dim = Config.BACKBONE_FILTERS * 2

        # Metadata Dim: 32
        self.meta_dim = 32

        # Total Fusion Dim: 512 + 512 + 256 + 32 = 1312
        fusion_dim = (
            self.flat_peak_dim
            + self.flat_shadow_dim
            + self.flat_global_dim
            + self.meta_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 1),
        )

    def forward(self, x_img, x_meta):
        # --- Backbone ---
        x = self.stage1(x_img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)  # Shape: (B, 256, 4, 4)

        # --- Decoupled Readout ---
        # Split the feature volume based on DualPooling origin
        # First 128 channels came from MaxPool (Peaks)
        # Last 128 channels came from MinPool (Shadows)
        x_peak_in = x[:, : Config.BACKBONE_FILTERS, :, :]
        x_shadow_in = x[:, Config.BACKBONE_FILTERS :, :, :]

        # Path A: Peak Processing
        x_peak = self.peak_conv(x_peak_in)
        x_peak = F.relu(x_peak)
        x_peak = x_peak.view(x_peak.size(0), -1)  # Flatten

        # Path B: Shadow Processing
        x_shadow = self.shadow_conv(x_shadow_in)
        x_shadow = F.relu(x_shadow)
        x_shadow = x_shadow.view(x_shadow.size(0), -1)  # Flatten

        # Path C: Global Statistics
        x_global = self.global_pool(x)
        x_global = x_global.view(x_global.size(0), -1)

        # --- Metadata Processing ---
        # Ensure correct shape (B, 1)
        if x_meta.dim() == 1:
            x_meta = x_meta.unsqueeze(1)
        x_m = self.meta_mlp(x_meta)

        # --- Fusion ---
        x_fused = torch.cat([x_peak, x_shadow, x_global, x_m], dim=1)

        # --- Classification ---
        out = self.classifier(x_fused)

        return out
