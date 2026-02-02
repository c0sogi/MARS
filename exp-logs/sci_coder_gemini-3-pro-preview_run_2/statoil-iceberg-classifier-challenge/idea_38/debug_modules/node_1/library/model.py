import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.model_components import ConvBlock, DualPooling, CBAM, SEBlock


class CRWBN(nn.Module):
    """
    Channel-Recalibrated Wide-Body Network (CR-WBN).

    Architecture:
    1. Visual Backbone: 4 Stages of Wide Conv -> CBAM -> DualPooling.
       - Uses "Sustained Width" (128 filters).
       - Uses "Delayed Integration" (256 -> 128 mapping in later blocks).
    2. Recalibrated Readout:
       - SE-Block for global context weighting.
       - Split paths: Spatial (Conv+Flatten) and Intensity (GAP).
    3. Metadata Branch: MLP for incidence angle.
    4. Fusion Head: Concatenation + Dense + Dropout.
    """

    def __init__(self):
        super(CRWBN, self).__init__()

        # ==========================
        # 1. Visual Backbone
        # ==========================

        # Stage 1: Input (3) -> 128 -> Pool(256)
        self.conv1 = ConvBlock(Config.INPUT_CHANNELS, Config.BACKBONE_FILTERS)
        self.cbam1 = CBAM(Config.BACKBONE_FILTERS)
        self.pool1 = DualPooling()  # Output channels: 128 * 2 = 256

        # Stage 2: Input (256) -> 128 -> Pool(256)
        # Delayed Integration: Compress 256 input channels to 128 filters
        self.conv2 = ConvBlock(Config.BACKBONE_FILTERS * 2, Config.BACKBONE_FILTERS)
        self.cbam2 = CBAM(Config.BACKBONE_FILTERS)
        self.pool2 = DualPooling()

        # Stage 3: Input (256) -> 128 -> Pool(256)
        self.conv3 = ConvBlock(Config.BACKBONE_FILTERS * 2, Config.BACKBONE_FILTERS)
        self.cbam3 = CBAM(Config.BACKBONE_FILTERS)
        self.pool3 = DualPooling()

        # Stage 4: Input (256) -> 128 -> Pool(256)
        self.conv4 = ConvBlock(Config.BACKBONE_FILTERS * 2, Config.BACKBONE_FILTERS)
        self.cbam4 = CBAM(Config.BACKBONE_FILTERS)
        self.pool4 = DualPooling()

        # ==========================
        # 2. Recalibrated Readout
        # ==========================

        # Global Recalibration (SE-Block) on the final 256-channel volume
        self.se_block = SEBlock(Config.BACKBONE_FILTERS * 2)  # 256 channels

        # Path A: Spatial Context
        # Compress 256 -> 64 channels, keeping spatial dim (4x4)
        self.spatial_conv = nn.Conv2d(
            Config.BACKBONE_FILTERS * 2,
            Config.READOUT_SPATIAL_DIM,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(Config.READOUT_SPATIAL_DIM)
        self.spatial_relu = nn.ReLU(inplace=True)

        # Calculate flattened dimension:
        # Input 75x75 -> Poolx4 -> 4x4 spatial size
        # 4 * 4 * 64 = 1024
        self.flat_dim = 4 * 4 * Config.READOUT_SPATIAL_DIM

        # Path B: Robust Intensity
        # Global Average Pooling on 256 channels
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.intensity_dim = Config.BACKBONE_FILTERS * 2

        # ==========================
        # 3. Metadata Branch
        # ==========================
        self.meta_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.meta_dim = 32

        # ==========================
        # 4. Fusion Head
        # ==========================
        fusion_input_dim = self.flat_dim + self.intensity_dim + self.meta_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 1),  # Output Logits
        )

    def forward(self, x_img, x_angle):
        # --- Visual Backbone ---
        # Stage 1
        x = self.conv1(x_img)
        x = self.cbam1(x)
        x = self.pool1(x)

        # Stage 2
        x = self.conv2(x)
        x = self.cbam2(x)
        x = self.pool2(x)

        # Stage 3
        x = self.conv3(x)
        x = self.cbam3(x)
        x = self.pool3(x)

        # Stage 4
        x = self.conv4(x)
        x = self.cbam4(x)
        x = self.pool4(x)

        # --- Recalibration ---
        x = self.se_block(x)

        # --- Readout ---
        # Path A: Spatial
        x_sp = self.spatial_conv(x)
        x_sp = self.spatial_bn(x_sp)
        x_sp = self.spatial_relu(x_sp)
        x_sp = x_sp.view(x_sp.size(0), -1)  # Flatten

        # Path B: Intensity
        x_int = self.gap(x).view(x.size(0), -1)

        # --- Metadata ---
        # Ensure correct shape (Batch, 1)
        if x_angle.dim() == 1:
            x_angle = x_angle.view(-1, 1)
        x_meta = self.meta_mlp(x_angle)

        # --- Fusion ---
        x_fused = torch.cat([x_sp, x_int, x_meta], dim=1)

        # --- Classification ---
        logits = self.classifier(x_fused)

        return logits
