import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.model_parts import RepBlock, ASPP, CoordinateAttention


class RepCResUNetSR(nn.Module):
    """
    Reparameterized Coordinate ResUNet with Skip Refinement (Rep-CResUNet-SR).

    Architecture:
    - Encoder: RepBlock stages with MaxPool downsampling.
    - Skip Connections: Refined using Coordinate Attention (CA).
    - Bottleneck: Atrous Spatial Pyramid Pooling (ASPP).
    - Decoder: Transposed Convolutions + RepBlocks.
    - Global Residual: Predicts noise residual (Output = Input - Noise).
    """

    def __init__(self):
        super(RepCResUNetSR, self).__init__()

        filters = Config.BASE_FILTERS  # 64

        # --- Encoder ---
        # Stage 1: 1 -> 64
        self.enc1 = RepBlock(in_channels=1, out_channels=filters)
        self.ca1 = CoordinateAttention(in_channels=filters)

        # Stage 2: 64 -> 128
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = RepBlock(in_channels=filters, out_channels=filters * 2)
        self.ca2 = CoordinateAttention(in_channels=filters * 2)

        # Stage 3: 128 -> 256
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = RepBlock(in_channels=filters * 2, out_channels=filters * 4)
        self.ca3 = CoordinateAttention(in_channels=filters * 4)

        # Stage 4: 256 -> 512
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = RepBlock(in_channels=filters * 4, out_channels=filters * 8)
        self.ca4 = CoordinateAttention(in_channels=filters * 8)

        # --- Bottleneck ---
        # 512 -> 1024
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ASPP(in_channels=filters * 8, out_channels=filters * 16)

        # --- Decoder ---
        # Stage 4: 1024 -> 512
        self.up4 = nn.ConvTranspose2d(
            filters * 16, filters * 8, kernel_size=2, stride=2
        )
        # Input: Upsampled (512) + Skip (512) = 1024 -> Output: 512
        self.dec4 = RepBlock(in_channels=filters * 16, out_channels=filters * 8)

        # Stage 3: 512 -> 256
        self.up3 = nn.ConvTranspose2d(filters * 8, filters * 4, kernel_size=2, stride=2)
        # Input: Upsampled (256) + Skip (256) = 512 -> Output: 256
        self.dec3 = RepBlock(in_channels=filters * 8, out_channels=filters * 4)

        # Stage 2: 256 -> 128
        self.up2 = nn.ConvTranspose2d(filters * 4, filters * 2, kernel_size=2, stride=2)
        # Input: Upsampled (128) + Skip (128) = 256 -> Output: 128
        self.dec2 = RepBlock(in_channels=filters * 4, out_channels=filters * 2)

        # Stage 1: 128 -> 64
        self.up1 = nn.ConvTranspose2d(filters * 2, filters, kernel_size=2, stride=2)
        # Input: Upsampled (64) + Skip (64) = 128 -> Output: 64
        self.dec1 = RepBlock(in_channels=filters * 2, out_channels=filters)

        # --- Output Head ---
        # Maps 64 features to 1 channel (Noise Residual)
        self.final_conv = nn.Conv2d(filters, 1, kernel_size=1)

    def forward(self, x):
        # --- Encoder Pass ---
        # Stage 1
        e1 = self.enc1(x)
        e1_skip = self.ca1(e1)  # Refine skip

        # Stage 2
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        e2_skip = self.ca2(e2)  # Refine skip

        # Stage 3
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        e3_skip = self.ca3(e3)  # Refine skip

        # Stage 4
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        e4_skip = self.ca4(e4)  # Refine skip

        # --- Bottleneck ---
        p4 = self.pool4(e4)
        b = self.bottleneck(p4)

        # --- Decoder Pass ---
        # Stage 4
        d4 = self.up4(b)
        # Concatenate with refined skip connection
        if d4.size() != e4_skip.size():
            d4 = F.interpolate(
                d4, size=e4_skip.shape[-2:], mode="bilinear", align_corners=False
            )
        d4 = torch.cat([d4, e4_skip], dim=1)
        d4 = self.dec4(d4)

        # Stage 3
        d3 = self.up3(d4)
        if d3.size() != e3_skip.size():
            d3 = F.interpolate(
                d3, size=e3_skip.shape[-2:], mode="bilinear", align_corners=False
            )
        d3 = torch.cat([d3, e3_skip], dim=1)
        d3 = self.dec3(d3)

        # Stage 2
        d2 = self.up2(d3)
        if d2.size() != e2_skip.size():
            d2 = F.interpolate(
                d2, size=e2_skip.shape[-2:], mode="bilinear", align_corners=False
            )
        d2 = torch.cat([d2, e2_skip], dim=1)
        d2 = self.dec2(d2)

        # Stage 1
        d1 = self.up1(d2)
        if d1.size() != e1_skip.size():
            d1 = F.interpolate(
                d1, size=e1_skip.shape[-2:], mode="bilinear", align_corners=False
            )
        d1 = torch.cat([d1, e1_skip], dim=1)
        d1 = self.dec1(d1)

        # --- Global Residual Learning ---
        # Predict noise
        noise_pred = self.final_conv(d1)

        # Clean image = Input - Noise
        clean_pred = x - noise_pred

        return clean_pred

    def switch_to_deploy(self):
        """
        Switches the model to deployment mode by fusing the reparameterizable blocks.
        This reduces computational cost during inference.
        """
        for m in self.modules():
            if isinstance(m, RepBlock):
                m.switch_to_deploy()
