import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.layers import ResBlock, ASPP, AttentionGate, DeepSupervisionHead


class DS_AG_CAC_ResUNet(nn.Module):
    """
    Deeply Supervised Attention-Gated Coordinate ResUNet (DS-AG-CAC-ResUNet).

    Architecture:
    - Encoder: ResNet-like blocks with Coordinate Attention.
    - Bottleneck: Atrous Spatial Pyramid Pooling (ASPP).
    - Decoder: Transposed Convolutions + Attention Gates + ResBlocks.
    - Deep Supervision: Auxiliary heads at intermediate decoder levels.
    """

    def __init__(self):
        super(DS_AG_CAC_ResUNet, self).__init__()

        self.use_ds = Config.USE_DEEP_SUPERVISION
        self.use_ag = Config.USE_ATTENTION_GATES
        self.use_ca = Config.USE_COORDINATE_ATTENTION
        self.use_aspp = Config.USE_ASPP

        filters = Config.BASE_FILTERS  # e.g., 64

        # --- Encoder ---
        # Initial Conv
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.SiLU(),
        )

        # Stage 1
        self.enc1 = ResBlock(filters, filters, use_ca=self.use_ca)
        self.pool1 = nn.MaxPool2d(2)

        # Stage 2
        self.enc2 = ResBlock(filters, filters * 2, use_ca=self.use_ca)
        self.pool2 = nn.MaxPool2d(2)

        # Stage 3
        self.enc3 = ResBlock(filters * 2, filters * 4, use_ca=self.use_ca)
        self.pool3 = nn.MaxPool2d(2)

        # Stage 4
        self.enc4 = ResBlock(filters * 4, filters * 8, use_ca=self.use_ca)
        self.pool4 = nn.MaxPool2d(2)

        # --- Bottleneck ---
        # If ASPP is enabled, use it; otherwise use a standard ResBlock
        if self.use_aspp:
            self.bottleneck = ASPP(filters * 8, filters * 16)
        else:
            self.bottleneck = ResBlock(filters * 8, filters * 16, use_ca=self.use_ca)

        # --- Decoder ---
        # Stage 4 (Bottleneck -> Dec4)
        # Input: Bottleneck (1024) -> Up (512) + Skip (512) -> Concat (1024) -> ResBlock (512)
        self.up4 = nn.ConvTranspose2d(
            filters * 16, filters * 8, kernel_size=2, stride=2
        )
        if self.use_ag:
            self.ag4 = AttentionGate(
                F_g=filters * 8, F_l=filters * 8, F_int=filters * 4
            )
        self.dec4 = ResBlock(filters * 16, filters * 8, use_ca=self.use_ca)
        if self.use_ds:
            self.ds_head4 = DeepSupervisionHead(filters * 8, 1)

        # Stage 3
        # Input: Dec4 (512) -> Up (256) + Skip (256) -> Concat (512) -> ResBlock (256)
        self.up3 = nn.ConvTranspose2d(filters * 8, filters * 4, kernel_size=2, stride=2)
        if self.use_ag:
            self.ag3 = AttentionGate(
                F_g=filters * 4, F_l=filters * 4, F_int=filters * 2
            )
        self.dec3 = ResBlock(filters * 8, filters * 4, use_ca=self.use_ca)
        if self.use_ds:
            self.ds_head3 = DeepSupervisionHead(filters * 4, 1)

        # Stage 2
        # Input: Dec3 (256) -> Up (128) + Skip (128) -> Concat (256) -> ResBlock (128)
        self.up2 = nn.ConvTranspose2d(filters * 4, filters * 2, kernel_size=2, stride=2)
        if self.use_ag:
            self.ag2 = AttentionGate(F_g=filters * 2, F_l=filters * 2, F_int=filters)
        self.dec2 = ResBlock(filters * 4, filters * 2, use_ca=self.use_ca)
        if self.use_ds:
            self.ds_head2 = DeepSupervisionHead(filters * 2, 1)

        # Stage 1
        # Input: Dec2 (128) -> Up (64) + Skip (64) -> Concat (128) -> ResBlock (64)
        self.up1 = nn.ConvTranspose2d(filters * 2, filters, kernel_size=2, stride=2)
        if self.use_ag:
            self.ag1 = AttentionGate(F_g=filters, F_l=filters, F_int=filters // 2)
        self.dec1 = ResBlock(filters * 2, filters, use_ca=self.use_ca)

        # --- Output ---
        self.final_conv = nn.Conv2d(filters, 1, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x1 = self.input_conv(x)
        x1 = self.enc1(x1)  # 64, H, W

        x2 = self.pool1(x1)
        x2 = self.enc2(x2)  # 128, H/2, W/2

        x3 = self.pool2(x2)
        x3 = self.enc3(x3)  # 256, H/4, W/4

        x4 = self.pool3(x3)
        x4 = self.enc4(x4)  # 512, H/8, W/8

        # --- Bottleneck ---
        x_b = self.pool4(x4)
        b = self.bottleneck(x_b)  # 1024, H/16, W/16

        # --- Decoder ---

        # Stage 4
        d4 = self.up4(b)  # 512, H/8, W/8

        if self.use_ag:
            skip4 = self.ag4(g=d4, x=x4)
        else:
            skip4 = x4

        d4 = torch.cat([d4, skip4], dim=1)  # 1024 channels
        d4 = self.dec4(d4)  # 512 channels

        # Stage 3
        d3 = self.up3(d4)  # 256, H/4, W/4

        if self.use_ag:
            skip3 = self.ag3(g=d3, x=x3)
        else:
            skip3 = x3

        d3 = torch.cat([d3, skip3], dim=1)  # 512 channels
        d3 = self.dec3(d3)  # 256 channels

        # Stage 2
        d2 = self.up2(d3)  # 128, H/2, W/2

        if self.use_ag:
            skip2 = self.ag2(g=d2, x=x2)
        else:
            skip2 = x2

        d2 = torch.cat([d2, skip2], dim=1)  # 256 channels
        d2 = self.dec2(d2)  # 128 channels

        # Stage 1
        d1 = self.up1(d2)  # 64, H, W

        if self.use_ag:
            skip1 = self.ag1(g=d1, x=x1)
        else:
            skip1 = x1

        d1 = torch.cat([d1, skip1], dim=1)  # 128 channels
        d1 = self.dec1(d1)  # 64 channels

        # --- Output ---
        final_output = self.final_conv(d1)

        outputs = [final_output]

        # Deep Supervision outputs
        # We return them if deep supervision is enabled.
        # The training loop is responsible for calculating loss on them.
        if self.use_ds:
            aux4 = self.ds_head4(d4)
            aux3 = self.ds_head3(d3)
            aux2 = self.ds_head2(d2)

            # Upsample auxiliary outputs to match input resolution for loss calculation
            # or keep them small and downsample ground truth.
            # Standard practice in many implementations is to upsample prediction to GT size.
            aux4 = F.interpolate(
                aux4, size=x.shape[2:], mode="bilinear", align_corners=False
            )
            aux3 = F.interpolate(
                aux3, size=x.shape[2:], mode="bilinear", align_corners=False
            )
            aux2 = F.interpolate(
                aux2, size=x.shape[2:], mode="bilinear", align_corners=False
            )

            outputs.extend([aux2, aux3, aux4])

        return outputs
