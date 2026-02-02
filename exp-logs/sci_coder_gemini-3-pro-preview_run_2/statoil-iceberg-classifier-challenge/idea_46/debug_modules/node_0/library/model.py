import torch
import torch.nn as nn
from library.config import Config
from library.model_components import WideBlock


class TripleStreamWideBodyNetwork(nn.Module):
    """
    Triple-Stream Wide-Body Network (TS-WBN).

    Architecture:
    1. Visual Branch: 4 Stages of WideBlocks with Triple-Stream Pooling.
       - Maintains high channel capacity (Wide-Body).
       - Propagates Max, Min, and Avg streams (Triple-Stream).
    2. Readout:
       - Path A: Spatial Context (Conv -> Flatten -> BN).
       - Path B: Robust Intensity (Global Avg Pool -> BN).
    3. Metadata Branch:
       - Deep Normalized Embedding for incidence angle.
    4. Fusion Head:
       - Concatenation -> Dense -> BN -> ReLU -> High Dropout -> Output.
    """

    def __init__(self):
        super(TripleStreamWideBodyNetwork, self).__init__()

        # ==========================
        # 1. Visual Branch (Backbone)
        # ==========================
        # Config.FILTERS is 128.
        # TripleStreamPooling expands output channels by 3x (128 * 3 = 384).

        # Stage 1: Input (3) -> WideBlock -> Output (384)
        # Input size: 75x75 -> Pool -> 37x37
        self.stage1 = WideBlock(
            in_channels=Config.NUM_CHANNELS, out_filters=Config.FILTERS
        )

        # Stage 2: Input (384) -> WideBlock -> Output (384)
        # Input size: 37x37 -> Pool -> 18x18
        self.stage2 = WideBlock(
            in_channels=Config.FILTERS * 3, out_filters=Config.FILTERS
        )

        # Stage 3: Input (384) -> WideBlock -> Output (384)
        # Input size: 18x18 -> Pool -> 9x9
        self.stage3 = WideBlock(
            in_channels=Config.FILTERS * 3, out_filters=Config.FILTERS
        )

        # Stage 4: Input (384) -> WideBlock -> Output (384)
        # Input size: 9x9 -> Pool -> 4x4
        self.stage4 = WideBlock(
            in_channels=Config.FILTERS * 3, out_filters=Config.FILTERS
        )

        # ==========================
        # 2. Readout Paths
        # ==========================

        # Path A: Spatial Context
        # Input: 384 x 4 x 4
        # Conv 3x3 -> 64 channels -> Flatten -> 1024 dim
        self.path_a_conv = nn.Conv2d(Config.FILTERS * 3, 64, kernel_size=3, padding=1)
        self.path_a_bn = nn.BatchNorm1d(64 * 4 * 4)  # 1024

        # Path B: Robust Intensity
        # Input: 384 x 4 x 4
        # Global Avg Pool -> 384 dim
        self.path_b_pool = nn.AdaptiveAvgPool2d(1)
        self.path_b_bn = nn.BatchNorm1d(Config.FILTERS * 3)  # 384

        # ==========================
        # 3. Metadata Branch
        # ==========================
        # Input: 1 scalar (incidence angle)
        meta_hidden_dim = 32
        self.meta_fc1 = nn.Linear(1, meta_hidden_dim)
        self.meta_relu1 = nn.ReLU(inplace=True)
        self.meta_fc2 = nn.Linear(meta_hidden_dim, meta_hidden_dim)
        self.meta_bn = nn.BatchNorm1d(meta_hidden_dim)
        self.meta_relu2 = nn.ReLU(inplace=True)

        # ==========================
        # 4. Fusion Head
        # ==========================
        # Concatenation size:
        # Path A (1024) + Path B (384) + Metadata (32) = 1440
        fusion_input_dim = (64 * 4 * 4) + (Config.FILTERS * 3) + meta_hidden_dim
        fusion_hidden_dim = 512

        self.fusion_fc = nn.Linear(fusion_input_dim, fusion_hidden_dim)
        self.fusion_bn = nn.BatchNorm1d(fusion_hidden_dim)
        self.fusion_relu = nn.ReLU(inplace=True)
        self.fusion_dropout = nn.Dropout(Config.DROPOUT)  # 0.5

        # Final Classification
        self.classifier = nn.Linear(fusion_hidden_dim, 1)

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image input of shape (B, 3, 75, 75)
            angle (torch.Tensor): Incidence angle input of shape (B,)
        """

        # --- Visual Branch ---
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # x shape is now (B, 384, 4, 4)

        # --- Readout Path A (Spatial) ---
        xa = self.path_a_conv(x)  # (B, 64, 4, 4)
        xa = xa.view(xa.size(0), -1)  # Flatten -> (B, 1024)
        xa = self.path_a_bn(xa)

        # --- Readout Path B (Intensity) ---
        xb = self.path_b_pool(x)  # (B, 384, 1, 1)
        xb = xb.view(xb.size(0), -1)  # Flatten -> (B, 384)
        xb = self.path_b_bn(xb)

        # --- Metadata Branch ---
        # Ensure angle is (B, 1)
        angle = angle.view(-1, 1)
        xm = self.meta_fc1(angle)
        xm = self.meta_relu1(xm)
        xm = self.meta_fc2(xm)
        xm = self.meta_bn(xm)
        xm = self.meta_relu2(xm)

        # --- Fusion ---
        # Concatenate all features
        feat = torch.cat([xa, xb, xm], dim=1)

        # Dense Head
        feat = self.fusion_fc(feat)
        feat = self.fusion_bn(feat)
        feat = self.fusion_relu(feat)
        feat = self.fusion_dropout(feat)

        # Classification
        out = self.classifier(feat)

        return out
