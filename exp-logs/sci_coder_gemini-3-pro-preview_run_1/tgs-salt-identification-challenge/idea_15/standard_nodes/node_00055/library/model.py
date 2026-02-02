import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.model_parts import ResidualBlock, DecoderBlock, CoordinateAttention


class DeepResUNet(nn.Module):
    """
    Deep Residual U-Net with Coordinate Attention and Deep Supervision.

    Architecture:
    - Encoder: ResNet-style blocks with [64, 128, 256, 512] filters.
    - Decoder: Upsampling blocks with Coordinate Attention [256, 128, 64, 32].
    - Deep Supervision: Auxiliary heads at 32x32 and 64x64 resolutions.
    """

    def __init__(self):
        super(DeepResUNet, self).__init__()

        # Configuration
        self.n_channels = Config.INPUT_CHANNELS
        self.enc_filters = Config.ENCODER_FILTERS  # [64, 128, 256, 512]
        self.dec_filters = Config.DECODER_FILTERS  # [256, 128, 64, 32]
        self.deep_supervision = Config.DEEP_SUPERVISION
        self.use_coord_att = Config.USE_COORD_ATTENTION

        # ---------------------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------------------
        # Initial Conv: (B, 2, 128, 128) -> (B, 64, 128, 128)
        self.conv1 = nn.Conv2d(
            self.n_channels, self.enc_filters[0], kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.enc_filters[0])
        self.relu = nn.ReLU(inplace=True)

        # Stage 1: 128x128
        self.enc1 = ResidualBlock(self.enc_filters[0], self.enc_filters[0], stride=1)

        # Stage 2: 64x64
        self.enc2 = ResidualBlock(self.enc_filters[0], self.enc_filters[1], stride=2)

        # Stage 3: 32x32
        self.enc3 = ResidualBlock(self.enc_filters[1], self.enc_filters[2], stride=2)

        # Stage 4: 16x16 (Bottleneck)
        self.enc4 = ResidualBlock(self.enc_filters[2], self.enc_filters[3], stride=2)

        # ---------------------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------------------
        # Dec 1: 16x16 -> 32x32 (Input: 512, Skip: 256 -> Out: 256)
        self.dec1 = DecoderBlock(
            self.enc_filters[3],
            self.enc_filters[2],
            self.dec_filters[0],
            use_coord_att=self.use_coord_att,
        )

        # Dec 2: 32x32 -> 64x64 (Input: 256, Skip: 128 -> Out: 128)
        self.dec2 = DecoderBlock(
            self.dec_filters[0],
            self.enc_filters[1],
            self.dec_filters[1],
            use_coord_att=self.use_coord_att,
        )

        # Dec 3: 64x64 -> 128x128 (Input: 128, Skip: 64 -> Out: 64)
        self.dec3 = DecoderBlock(
            self.dec_filters[1],
            self.enc_filters[0],
            self.dec_filters[2],
            use_coord_att=self.use_coord_att,
        )

        # Dec 4: Final Refinement 128x128 (Input: 64 -> Out: 32)
        # We build this manually because DecoderBlock enforces upsampling
        self.dec4_conv1 = nn.Conv2d(
            self.dec_filters[2],
            self.dec_filters[3],
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.dec4_bn1 = nn.BatchNorm2d(self.dec_filters[3])
        self.dec4_conv2 = nn.Conv2d(
            self.dec_filters[3],
            self.dec_filters[3],
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.dec4_bn2 = nn.BatchNorm2d(self.dec_filters[3])

        if self.use_coord_att:
            self.dec4_att = CoordinateAttention(self.dec_filters[3])

        # ---------------------------------------------------------------------
        # Heads
        # ---------------------------------------------------------------------
        # Final Head (128x128)
        self.final_head = nn.Conv2d(self.dec_filters[3], 1, kernel_size=1)

        if self.deep_supervision:
            # Aux Head 1 (from Dec 1 - 32x32)
            self.aux1_head = nn.Conv2d(self.dec_filters[0], 1, kernel_size=1)
            # Aux Head 2 (from Dec 2 - 64x64)
            self.aux2_head = nn.Conv2d(self.dec_filters[1], 1, kernel_size=1)

    def forward(self, x):
        # x shape: (B, 2, 128, 128)

        # --- Encoder ---
        x_in = self.relu(self.bn1(self.conv1(x)))  # (B, 64, 128, 128)

        e1 = self.enc1(x_in)  # (B, 64, 128, 128)
        e2 = self.enc2(e1)  # (B, 128, 64, 64)
        e3 = self.enc3(e2)  # (B, 256, 32, 32)
        e4 = self.enc4(e3)  # (B, 512, 16, 16)

        # --- Decoder ---
        d1 = self.dec1(e4, e3)  # (B, 256, 32, 32)
        d2 = self.dec2(d1, e2)  # (B, 128, 64, 64)
        d3 = self.dec3(d2, e1)  # (B, 64, 128, 128)

        # --- Final Refinement ---
        d4 = self.relu(self.dec4_bn1(self.dec4_conv1(d3)))
        d4 = self.relu(self.dec4_bn2(self.dec4_conv2(d4)))

        if self.use_coord_att:
            d4 = self.dec4_att(d4)  # (B, 32, 128, 128)

        # --- Heads ---
        logits = self.final_head(d4)

        if self.training and self.deep_supervision:
            # Compute aux logits
            logits_aux1 = self.aux1_head(d1)  # (B, 1, 32, 32)
            logits_aux2 = self.aux2_head(d2)  # (B, 1, 64, 64)

            # Upsample to input resolution for loss calculation
            logits_aux1 = F.interpolate(
                logits_aux1, size=x.shape[2:], mode="bilinear", align_corners=True
            )
            logits_aux2 = F.interpolate(
                logits_aux2, size=x.shape[2:], mode="bilinear", align_corners=True
            )

            # Return tuple: (Main, Aux_HighRes, Aux_LowRes)
            return logits, logits_aux2, logits_aux1

        return logits
