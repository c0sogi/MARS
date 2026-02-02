import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE) Module.
    Enhances meaningful features by recalibrating channel and spatial responses.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, in_channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ConvBlock(nn.Module):
    """
    Standard U-Net++ Decoder Block.
    Consists of two 3x3 convolutions with BatchNorm and ReLU, followed by scSE attention.
    """

    def __init__(self, in_channels, out_channels, use_scse=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.scse = SCSEModule(out_channels) if use_scse else nn.Identity()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.scse(x)
        return x


class SaltUNetPlusPlus(nn.Module):
    """
    U-Net++ with ResNeXt-50 Encoder and Deep Supervision.
    Designed for Salt Segmentation with lightweight decoder channels.
    """

    def __init__(self, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision

        # 1. Encoder (ResNeXt-50 32x4d)
        # in_chans=3 matches the [Seismic, Seismic, Depth] input strategy
        self.encoder = timm.create_model(
            Config.ENCODER_NAME,
            pretrained=True,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # Encoder channels: [64, 256, 512, 1024, 2048] for ResNeXt-50
        # Corresponds to Strides: [2, 4, 8, 16, 32]
        enc_ch = self.encoder.feature_info.channels()

        # Decoder Channel Configuration from Config
        # Config.DECODER_CHANNELS = (256, 128, 64, 32, 16)
        # We map these to levels: L4(256), L3(128), L2(64), L1(32), L0(16)
        ch_l4 = Config.DECODER_CHANNELS[0]  # 256
        ch_l3 = Config.DECODER_CHANNELS[1]  # 128
        ch_l2 = Config.DECODER_CHANNELS[2]  # 64
        ch_l1 = Config.DECODER_CHANNELS[3]  # 32
        ch_l0 = Config.DECODER_CHANNELS[4]  # 16

        # 2. Decoder Blocks (Nested U-Net Structure)

        # Level 4 (Stride 32) - Center Block
        # Projects encoder bottleneck to lower dimension
        self.conv4_0 = ConvBlock(enc_ch[4], ch_l4)

        # Level 3 (Stride 16)
        # x_3_1: [e3, U(x_4_0)]
        self.conv3_1 = ConvBlock(enc_ch[3] + ch_l4, ch_l3)

        # Level 2 (Stride 8)
        # x_2_1: [e2, U(e3)] - Standard U-Net connection
        self.conv2_1 = ConvBlock(enc_ch[2] + enc_ch[3], ch_l2)
        # x_2_2: [e2, x_2_1, U(x_3_1)]
        self.conv2_2 = ConvBlock(enc_ch[2] + ch_l2 + ch_l3, ch_l2)

        # Level 1 (Stride 4)
        # x_1_1: [e1, U(e2)]
        self.conv1_1 = ConvBlock(enc_ch[1] + enc_ch[2], ch_l1)
        # x_1_2: [e1, x_1_1, U(x_2_1)]
        self.conv1_2 = ConvBlock(enc_ch[1] + ch_l1 + ch_l2, ch_l1)
        # x_1_3: [e1, x_1_1, x_1_2, U(x_2_2)]
        self.conv1_3 = ConvBlock(enc_ch[1] + ch_l1 + ch_l1 + ch_l2, ch_l1)

        # Level 0 (Stride 2) - Output Level
        # x_0_1: [e0, U(e1)]
        self.conv0_1 = ConvBlock(enc_ch[0] + enc_ch[1], ch_l0)
        # x_0_2: [e0, x_0_1, U(x_1_1)]
        self.conv0_2 = ConvBlock(enc_ch[0] + ch_l0 + ch_l1, ch_l0)
        # x_0_3: [e0, x_0_1, x_0_2, U(x_1_2)]
        self.conv0_3 = ConvBlock(enc_ch[0] + ch_l0 + ch_l0 + ch_l1, ch_l0)
        # x_0_4: [e0, x_0_1, x_0_2, x_0_3, U(x_1_3)] - Final Node
        self.conv0_4 = ConvBlock(enc_ch[0] + ch_l0 + ch_l0 + ch_l0 + ch_l1, ch_l0)

        # 3. Segmentation Heads
        # Applied to L0 nodes (Stride 2)
        self.final_head = nn.Conv2d(ch_l0, 1, kernel_size=1)

        # Deep Supervision Heads
        self.head1 = nn.Conv2d(ch_l0, 1, kernel_size=1)
        self.head2 = nn.Conv2d(ch_l0, 1, kernel_size=1)
        self.head3 = nn.Conv2d(ch_l0, 1, kernel_size=1)

    def forward(self, x):
        # Input shape: (B, 3, H, W) e.g., (B, 3, 128, 128)
        input_size = x.shape[-2:]

        # Encoder Features
        # e0: Stride 2, e1: Stride 4, e2: Stride 8, e3: Stride 16, e4: Stride 32
        features = self.encoder(x)
        e0, e1, e2, e3, e4 = features

        # --- Decoder Forward Pass ---

        # Level 4
        x4_0 = self.conv4_0(e4)
        u4_0 = F.interpolate(x4_0, scale_factor=2, mode="bilinear", align_corners=True)

        # Level 3
        # x3_0 is e3
        x3_1 = self.conv3_1(torch.cat([e3, u4_0], dim=1))
        u3_1 = F.interpolate(x3_1, scale_factor=2, mode="bilinear", align_corners=True)
        # Upsample e3 for L2 connections
        u3_0 = F.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=True)

        # Level 2
        # x2_0 is e2
        x2_1 = self.conv2_1(torch.cat([e2, u3_0], dim=1))
        u2_1 = F.interpolate(x2_1, scale_factor=2, mode="bilinear", align_corners=True)

        x2_2 = self.conv2_2(torch.cat([e2, x2_1, u3_1], dim=1))
        u2_2 = F.interpolate(x2_2, scale_factor=2, mode="bilinear", align_corners=True)
        # Upsample e2 for L1 connections
        u2_0 = F.interpolate(e2, scale_factor=2, mode="bilinear", align_corners=True)

        # Level 1
        # x1_0 is e1
        x1_1 = self.conv1_1(torch.cat([e1, u2_0], dim=1))
        u1_1 = F.interpolate(x1_1, scale_factor=2, mode="bilinear", align_corners=True)

        x1_2 = self.conv1_2(torch.cat([e1, x1_1, u2_1], dim=1))
        u1_2 = F.interpolate(x1_2, scale_factor=2, mode="bilinear", align_corners=True)

        x1_3 = self.conv1_3(torch.cat([e1, x1_1, x1_2, u2_2], dim=1))
        u1_3 = F.interpolate(x1_3, scale_factor=2, mode="bilinear", align_corners=True)
        # Upsample e1 for L0 connections
        u1_0 = F.interpolate(e1, scale_factor=2, mode="bilinear", align_corners=True)

        # Level 0 (Output Level - Stride 2)
        # x0_0 is e0
        x0_1 = self.conv0_1(torch.cat([e0, u1_0], dim=1))
        x0_2 = self.conv0_2(torch.cat([e0, x0_1, u1_1], dim=1))
        x0_3 = self.conv0_3(torch.cat([e0, x0_1, x0_2, u1_2], dim=1))
        x0_4 = self.conv0_4(torch.cat([e0, x0_1, x0_2, x0_3, u1_3], dim=1))

        # --- Heads & Upsampling ---
        # All outputs are currently at Stride 2 (e.g., 64x64 for 128x128 input)
        # We need to upsample to input_size (128x128)

        final_logit = self.final_head(x0_4)
        final_logit = F.interpolate(
            final_logit, size=input_size, mode="bilinear", align_corners=True
        )

        if self.deep_supervision and self.training:
            # Calculate auxiliary logits
            logit1 = self.head1(x0_1)
            logit2 = self.head2(x0_2)
            logit3 = self.head3(x0_3)

            # Upsample auxiliaries
            logit1 = F.interpolate(
                logit1, size=input_size, mode="bilinear", align_corners=True
            )
            logit2 = F.interpolate(
                logit2, size=input_size, mode="bilinear", align_corners=True
            )
            logit3 = F.interpolate(
                logit3, size=input_size, mode="bilinear", align_corners=True
            )

            return [logit1, logit2, logit3, final_logit]

        return final_logit
