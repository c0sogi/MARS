import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import ModelConfig


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation Module.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Channel Squeeze & Excitation
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        # Spatial Squeeze & Excitation
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ConvBlock(nn.Module):
    """
    Standard Convolution Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> SCSE
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.scse = SCSEModule(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.scse(x)
        return x


class SaltUNetPlusPlus(nn.Module):
    """
    U-Net++ with ResNeXt-50 Encoder and Deep Supervision.
    """

    def __init__(self):
        super().__init__()

        # 1. Encoder (ResNeXt-50 32x4d)
        # We use timm to load the backbone.
        # in_chans=3 allows us to use the [Seismic, Seismic, Depth] input strategy directly.
        self.encoder = timm.create_model(
            ModelConfig.ENCODER, pretrained=True, features_only=True, in_chans=3
        )

        # Encoder Feature Channels (e.g., [64, 256, 512, 1024, 2048])
        e_ch = self.encoder.feature_info.channels()

        # Decoder Filter Widths
        d_ch = [32, 64, 128, 256]  # L0, L1, L2, L3

        # 2. Decoder Blocks (Dense Connections)
        # Naming convention: conv_L_j where L is level (row) and j is dense block index (column)

        # --- Column j=1 (Depends on Encoder j=0) ---
        # x_0_1: up(x_1_0) + x_0_0
        self.up_1_0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_0_1 = ConvBlock(e_ch[1] + e_ch[0], d_ch[0])

        # x_1_1: up(x_2_0) + x_1_0
        self.up_2_0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_1_1 = ConvBlock(e_ch[2] + e_ch[1], d_ch[1])

        # x_2_1: up(x_3_0) + x_2_0
        self.up_3_0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_2_1 = ConvBlock(e_ch[3] + e_ch[2], d_ch[2])

        # x_3_1: up(x_4_0) + x_3_0
        self.up_4_0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_3_1 = ConvBlock(e_ch[4] + e_ch[3], d_ch[3])

        # --- Column j=2 ---
        # x_0_2: up(x_1_1) + x_0_0 + x_0_1
        self.up_1_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_0_2 = ConvBlock(d_ch[1] + e_ch[0] + d_ch[0], d_ch[0])

        # x_1_2: up(x_2_1) + x_1_0 + x_1_1
        self.up_2_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_1_2 = ConvBlock(d_ch[2] + e_ch[1] + d_ch[1], d_ch[1])

        # x_2_2: up(x_3_1) + x_2_0 + x_2_1
        self.up_3_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_2_2 = ConvBlock(d_ch[3] + e_ch[2] + d_ch[2], d_ch[2])

        # --- Column j=3 ---
        # x_0_3: up(x_1_2) + x_0_0 + x_0_1 + x_0_2
        self.up_1_2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_0_3 = ConvBlock(d_ch[1] + e_ch[0] + d_ch[0] * 2, d_ch[0])

        # x_1_3: up(x_2_2) + x_1_0 + x_1_1 + x_1_2
        self.up_2_2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_1_3 = ConvBlock(d_ch[2] + e_ch[1] + d_ch[1] * 2, d_ch[1])

        # --- Column j=4 ---
        # x_0_4: up(x_1_3) + x_0_0 + x_0_1 + x_0_2 + x_0_3
        self.up_1_3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_0_4 = ConvBlock(d_ch[1] + e_ch[0] + d_ch[0] * 3, d_ch[0])

        # 3. Final Segmentation Heads (Deep Supervision)
        # We produce an output for every node at Level 0
        self.final_1 = nn.Conv2d(d_ch[0], 1, 1)
        self.final_2 = nn.Conv2d(d_ch[0], 1, 1)
        self.final_3 = nn.Conv2d(d_ch[0], 1, 1)
        self.final_4 = nn.Conv2d(d_ch[0], 1, 1)

    def forward(self, x):
        # x shape: (B, 3, 128, 128)

        # --- Encoder ---
        enc_feats = self.encoder(x)
        x_0_0 = enc_feats[0]  # Stride 2 (64x64)
        x_1_0 = enc_feats[1]  # Stride 4 (32x32)
        x_2_0 = enc_feats[2]  # Stride 8 (16x16)
        x_3_0 = enc_feats[3]  # Stride 16 (8x8)
        x_4_0 = enc_feats[4]  # Stride 32 (4x4)

        # --- Decoder Column j=1 ---
        x_3_1 = self.conv_3_1(torch.cat([self.up_4_0(x_4_0), x_3_0], dim=1))
        x_2_1 = self.conv_2_1(torch.cat([self.up_3_0(x_3_0), x_2_0], dim=1))
        x_1_1 = self.conv_1_1(torch.cat([self.up_2_0(x_2_0), x_1_0], dim=1))
        x_0_1 = self.conv_0_1(torch.cat([self.up_1_0(x_1_0), x_0_0], dim=1))

        # --- Decoder Column j=2 ---
        x_2_2 = self.conv_2_2(torch.cat([self.up_3_1(x_3_1), x_2_0, x_2_1], dim=1))
        x_1_2 = self.conv_1_2(torch.cat([self.up_2_1(x_2_1), x_1_0, x_1_1], dim=1))
        x_0_2 = self.conv_0_2(torch.cat([self.up_1_1(x_1_1), x_0_0, x_0_1], dim=1))

        # --- Decoder Column j=3 ---
        x_1_3 = self.conv_1_3(
            torch.cat([self.up_2_2(x_2_2), x_1_0, x_1_1, x_1_2], dim=1)
        )
        x_0_3 = self.conv_0_3(
            torch.cat([self.up_1_2(x_1_2), x_0_0, x_0_1, x_0_2], dim=1)
        )

        # --- Decoder Column j=4 ---
        x_0_4 = self.conv_0_4(
            torch.cat([self.up_1_3(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], dim=1)
        )

        # --- Output Heads ---
        # All x_0_j are 64x64. We project to 1 channel and upsample to 128x128.
        out_1 = F.interpolate(
            self.final_1(x_0_1), scale_factor=2, mode="bilinear", align_corners=True
        )
        out_2 = F.interpolate(
            self.final_2(x_0_2), scale_factor=2, mode="bilinear", align_corners=True
        )
        out_3 = F.interpolate(
            self.final_3(x_0_3), scale_factor=2, mode="bilinear", align_corners=True
        )
        out_4 = F.interpolate(
            self.final_4(x_0_4), scale_factor=2, mode="bilinear", align_corners=True
        )

        if self.training:
            # Return list for Deep Supervision Loss
            return [out_1, out_2, out_3, out_4]
        else:
            # Return only the most refined output for inference/validation
            return out_4
