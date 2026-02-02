import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.custom_layers import DualPooling, CBAM, TriPathReadout


class TriPathWideBodyNet(nn.Module):
    """
    Tri-Path Wide-Body Network (TP-WBN).

    A specialized architecture for SAR image classification that combines:
    1. A Wide-Body Delayed-Integration Backbone to maintain signal capacity.
    2. A Tri-Path Readout to explicitly model spatial structure, signal intensity, and texture variance.
    3. A metadata branch for incidence angle integration.
    """

    def __init__(self):
        super(TriPathWideBodyNet, self).__init__()

        # Hyperparameters
        self.backbone_filters = Config.BACKBONE_FILTERS  # 128
        self.input_channels = Config.INPUT_CHANNELS  # 3
        self.dropout_rate = Config.DROPOUT_RATE  # 0.5

        # ==========================================
        # 1. Visual Branch (Wide-Body Backbone)
        # ==========================================

        # Block 1: Input (3) -> Conv(128) -> ... -> DualPool(256)
        self.block1_conv = nn.Conv2d(
            self.input_channels,
            self.backbone_filters,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.block1_bn = nn.BatchNorm2d(self.backbone_filters)
        self.block1_cbam = CBAM(self.backbone_filters)
        self.block1_pool = DualPooling(kernel_size=2, stride=2)

        # Block 2: Input(256) -> Conv(128) -> ... -> DualPool(256)
        # Input channels are 256 because DualPooling concatenates Max and Min (128 + 128)
        self.block2_conv = nn.Conv2d(
            self.backbone_filters * 2,
            self.backbone_filters,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.block2_bn = nn.BatchNorm2d(self.backbone_filters)
        self.block2_cbam = CBAM(self.backbone_filters)
        self.block2_pool = DualPooling(kernel_size=2, stride=2)

        # Block 3: Input(256) -> Conv(128) -> ... -> DualPool(256)
        self.block3_conv = nn.Conv2d(
            self.backbone_filters * 2,
            self.backbone_filters,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.block3_bn = nn.BatchNorm2d(self.backbone_filters)
        self.block3_cbam = CBAM(self.backbone_filters)
        self.block3_pool = DualPooling(kernel_size=2, stride=2)

        # Block 4: Input(256) -> Conv(128) -> ... -> DualPool(256)
        self.block4_conv = nn.Conv2d(
            self.backbone_filters * 2,
            self.backbone_filters,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.block4_bn = nn.BatchNorm2d(self.backbone_filters)
        self.block4_cbam = CBAM(self.backbone_filters)
        self.block4_pool = DualPooling(kernel_size=2, stride=2)

        # ==========================================
        # 2. Tri-Path Readout Interface
        # ==========================================
        # Input: 256 channels (from Block 4 DualPooling)
        # Spatial Dimensions: 75 -> 37 -> 18 -> 9 -> 4
        # Path A (Spatial): 48 channels * 4 * 4 = 768
        # Path B (Intensity): 256 channels (Global Avg)
        # Path C (Texture): 256 channels (Global Std)
        # Total Visual Vector: 768 + 256 + 256 = 1280
        self.readout = TriPathReadout(
            in_channels=self.backbone_filters * 2, path_a_out_channels=48
        )
        self.visual_dim = 1280

        # ==========================================
        # 3. Metadata Branch
        # ==========================================
        self.meta_dim = 64
        self.meta_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.meta_dim),
            nn.BatchNorm1d(self.meta_dim),
            nn.ReLU(inplace=True),
        )

        # ==========================================
        # 4. Fusion Head
        # ==========================================
        fusion_input_dim = self.visual_dim + self.meta_dim  # 1280 + 64 = 1344
        dense_units = 512

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, dense_units),
            nn.BatchNorm1d(dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(dense_units, 1),
        )

    def forward(self, x_img, x_inc):
        """
        Forward pass of the TP-WBN.
        x_img: Tensor of shape (Batch, 3, 75, 75)
        x_inc: Tensor of shape (Batch,) or (Batch, 1)
        """

        # --- Visual Branch ---
        # Block 1
        x = self.block1_conv(x_img)
        x = self.block1_bn(x)
        x = F.relu(x)
        x = self.block1_cbam(x)
        x = self.block1_pool(x)

        # Block 2
        x = self.block2_conv(x)
        x = self.block2_bn(x)
        x = F.relu(x)
        x = self.block2_cbam(x)
        x = self.block2_pool(x)

        # Block 3
        x = self.block3_conv(x)
        x = self.block3_bn(x)
        x = F.relu(x)
        x = self.block3_cbam(x)
        x = self.block3_pool(x)

        # Block 4
        x = self.block4_conv(x)
        x = self.block4_bn(x)
        x = F.relu(x)
        x = self.block4_cbam(x)
        x = self.block4_pool(x)

        # Readout (Tri-Path)
        visual_vec = self.readout(x)

        # --- Metadata Branch ---
        # Ensure correct shape for MLP
        if x_inc.dim() == 1:
            x_inc = x_inc.unsqueeze(1)

        meta_vec = self.meta_mlp(x_inc)

        # --- Fusion ---
        combined = torch.cat([visual_vec, meta_vec], dim=1)
        logits = self.fusion_head(combined)

        return logits
