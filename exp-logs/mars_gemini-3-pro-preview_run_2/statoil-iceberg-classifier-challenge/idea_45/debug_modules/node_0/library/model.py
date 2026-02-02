import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.layers import CBAM, DualPooling, ContextGating


class VisualBackbone(nn.Module):
    """
    Wide-Body Delayed-Integration Backbone.
    Consists of 4 stages with sustained width, CBAM attention, and Dual-Stream Pooling.
    """

    def __init__(self):
        super(VisualBackbone, self).__init__()

        self.filters = Config.BACKBONE_FILTERS  # 128

        # --- Stage 1 ---
        # Input: (B, 3, 75, 75)
        # Conv: 3 -> 128
        self.conv1 = nn.Conv2d(
            Config.NUM_CHANNELS, self.filters, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.filters)
        self.cbam1 = CBAM(self.filters)
        self.pool1 = DualPooling(kernel_size=2, stride=2)  # Out: 128*2 = 256 channels

        # --- Stage 2 ---
        # Input: (B, 256, 37, 37)
        # Delayed Integration: 256 -> 128
        self.conv2 = nn.Conv2d(
            self.filters * 2, self.filters, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(self.filters)
        self.cbam2 = CBAM(self.filters)
        self.pool2 = DualPooling(kernel_size=2, stride=2)  # Out: 256 channels

        # --- Stage 3 ---
        # Input: (B, 256, 18, 18)
        # Delayed Integration: 256 -> 128
        self.conv3 = nn.Conv2d(
            self.filters * 2, self.filters, kernel_size=3, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.filters)
        self.cbam3 = CBAM(self.filters)
        self.pool3 = DualPooling(kernel_size=2, stride=2)  # Out: 256 channels

        # --- Stage 4 ---
        # Input: (B, 256, 9, 9)
        # Delayed Integration: 256 -> 128
        self.conv4 = nn.Conv2d(
            self.filters * 2, self.filters, kernel_size=3, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(self.filters)
        self.cbam4 = CBAM(self.filters)
        self.pool4 = DualPooling(kernel_size=2, stride=2)  # Out: 256 channels
        # Final Output: (B, 256, 4, 4)

    def forward(self, x):
        # Stage 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)
        x = self.pool1(x)

        # Stage 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        x = self.pool2(x)

        # Stage 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.cbam3(x)
        x = self.pool3(x)

        # Stage 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.cbam4(x)
        x = self.pool4(x)

        return x


class MetadataBranch(nn.Module):
    """
    Deep Normalized Embedding for Incidence Angle.
    Structure: Linear -> ReLU -> Linear -> BN -> ReLU
    """

    def __init__(self, output_dim=32):
        super(MetadataBranch, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class CGWBN(nn.Module):
    """
    Context-Gated Wide-Body Network.
    Integrates Visual Backbone, Context-Gated Readout, and Metadata Branch.
    """

    def __init__(self):
        super(CGWBN, self).__init__()

        # 1. Visual Branch
        self.backbone = VisualBackbone()

        # Backbone output: 256 channels, 4x4 spatial
        self.backbone_channels = Config.BACKBONE_FILTERS * 2  # 256
        self.spatial_size = 4

        # 2. Context-Gated Dual-Path Readout
        # Path A: Global Context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.bn_context = nn.BatchNorm1d(self.backbone_channels)

        # Path B: Spatial Features
        # Compress 256 -> 64 channels before flattening
        self.spatial_compress = nn.Conv2d(
            self.backbone_channels, 64, kernel_size=3, padding=1, bias=False
        )
        self.spatial_flat_dim = (
            64 * self.spatial_size * self.spatial_size
        )  # 64 * 4 * 4 = 1024

        # Gating Mechanism
        self.gating = ContextGating(
            context_dim=self.backbone_channels, feature_dim=self.spatial_flat_dim
        )
        self.bn_spatial = nn.BatchNorm1d(self.spatial_flat_dim)

        # 3. Metadata Branch
        self.meta_dim = 32
        self.meta_branch = MetadataBranch(output_dim=self.meta_dim)

        # 4. Fusion Head
        # Inputs: Gated Spatial (1024) + Global Context (256) + Metadata (32)
        fusion_input_dim = (
            self.spatial_flat_dim + self.backbone_channels + self.meta_dim
        )

        self.dense_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 1),
        )

    def forward(self, img, inc_angle):
        # --- Visual Branch ---
        features = self.backbone(img)  # (B, 256, 4, 4)

        # --- Readout Path A: Global Context ---
        # (B, 256, 1, 1) -> (B, 256)
        global_context = self.global_pool(features).view(features.size(0), -1)

        # --- Readout Path B: Spatial Features ---
        # (B, 256, 4, 4) -> (B, 64, 4, 4) -> (B, 1024)
        spatial_feat = self.spatial_compress(features)
        spatial_feat = spatial_feat.view(spatial_feat.size(0), -1)

        # --- Context Gating ---
        # Modulate spatial features based on global context
        gated_spatial = self.gating(spatial_feat, global_context)

        # Normalize Readouts
        global_context = self.bn_context(global_context)
        gated_spatial = self.bn_spatial(gated_spatial)

        # --- Metadata Branch ---
        meta_feat = self.meta_branch(inc_angle)

        # --- Fusion ---
        fused = torch.cat([gated_spatial, global_context, meta_feat], dim=1)
        logits = self.dense_head(fused)

        return logits
