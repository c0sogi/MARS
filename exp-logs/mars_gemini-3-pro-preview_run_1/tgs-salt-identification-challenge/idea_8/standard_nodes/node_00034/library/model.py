import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.model_components import ResidualBlock, DecoderBlock


class DeepResUNet(nn.Module):
    """
    Deep Residual U-Net with Deep Supervision (No ASPP).
    Optimized for small input resolutions (101x101).

    Architecture:
    1. Encoder: Deep Residual Backbone (64 -> 128 -> 256 -> 512 filters).
    2. Bottleneck: Identity (Direct connection).
    3. Decoder: Bilinear Upsampling + scSE Attention + Skip Connections.
    4. Heads: Main prediction head + Auxiliary heads for deep supervision.
    """

    def __init__(
        self,
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        filters=Config.ENCODER_FILTERS,
    ):
        super(DeepResUNet, self).__init__()

        # ---------------------------------------------------------------------
        # Encoder (Deep Residual Backbone)
        # ---------------------------------------------------------------------
        # Initial convolution to expand input channels (2 -> 64)
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
        )

        # Level 1: 128x128, 64 filters
        self.enc1 = ResidualBlock(filters, filters)

        # Level 2: 64x64, 128 filters (Downsampled via Stride 2)
        self.enc2 = ResidualBlock(filters, filters * 2, stride=2)

        # Level 3: 32x32, 256 filters (Downsampled via Stride 2)
        self.enc3 = ResidualBlock(filters * 2, filters * 4, stride=2)

        # Level 4: 16x16, 512 filters (Downsampled via Stride 2)
        self.enc4 = ResidualBlock(filters * 4, filters * 8, stride=2)

        # ---------------------------------------------------------------------
        # Decoder (with scSE Attention)
        # ---------------------------------------------------------------------
        # Decoder 3: Upsample 16x16 -> 32x32
        # Input: 512 (Enc4), Skip: 256 (Enc3) -> Output: 256
        self.dec3 = DecoderBlock(filters * 8, filters * 4, filters * 4)

        # Decoder 2: Upsample 32x32 -> 64x64
        # Input: 256 (Dec3), Skip: 128 (Enc2) -> Output: 128
        self.dec2 = DecoderBlock(filters * 4, filters * 2, filters * 2)

        # Decoder 1: Upsample 64x64 -> 128x128
        # Input: 128 (Dec2), Skip: 64 (Enc1) -> Output: 64
        self.dec1 = DecoderBlock(filters * 2, filters, filters)

        # ---------------------------------------------------------------------
        # Prediction Heads
        # ---------------------------------------------------------------------
        # Main Head (Final 128x128 prediction)
        self.final_conv = nn.Conv2d(filters, out_channels, kernel_size=1)

        # Auxiliary Heads for Deep Supervision
        # Attached to Decoder 3 (32x32) and Decoder 2 (64x64)
        self.aux_head3 = nn.Conv2d(filters * 4, out_channels, kernel_size=1)
        self.aux_head2 = nn.Conv2d(filters * 2, out_channels, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ---------------------------------------------------------------------
        # Encoder Forward Pass
        # ---------------------------------------------------------------------
        x0 = self.input_conv(x)  # (B, 64, 128, 128)
        x1 = self.enc1(x0)  # (B, 64, 128, 128)
        x2 = self.enc2(x1)  # (B, 128, 64, 64)
        x3 = self.enc3(x2)  # (B, 256, 32, 32)
        x4 = self.enc4(x3)  # (B, 512, 16, 16)

        # ---------------------------------------------------------------------
        # Decoder Forward Pass
        # ---------------------------------------------------------------------
        # Skip ASPP, pass x4 directly
        d3 = self.dec3(x4, x3)  # (B, 256, 32, 32)
        d2 = self.dec2(d3, x2)  # (B, 128, 64, 64)
        d1 = self.dec1(d2, x1)  # (B, 64, 128, 128)

        # ---------------------------------------------------------------------
        # Heads Forward Pass
        # ---------------------------------------------------------------------
        logits = self.final_conv(d1)

        if self.training:
            # Deep Supervision: Return auxiliary logits upsampled to input size
            # Aux 3 (from 32x32)
            aux3 = self.aux_head3(d3)
            aux3 = F.interpolate(
                aux3, size=logits.shape[2:], mode="bilinear", align_corners=False
            )

            # Aux 2 (from 64x64)
            aux2 = self.aux_head2(d2)
            aux2 = F.interpolate(
                aux2, size=logits.shape[2:], mode="bilinear", align_corners=False
            )

            return logits, aux2, aux3

        return logits
