import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE) Module.
    Enhances important features by recalibrating both spatial and channel dimensions.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel Squeeze and Excitation
        # Use max(1, ...) to ensure reduction doesn't result in 0 channels
        reduced_channels = max(1, channels // reduction)
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid(),
        )
        # Spatial Squeeze and Excitation
        self.sSE = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # Concurrent: combine channel attention and spatial attention
        return x * self.cSE(x) + x * self.sSE(x)


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for U-Net++ nodes.
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> SCSE
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.scse = SCSEModule(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.scse(x)
        return x


class ResNeXt50UNetPlusPlus(nn.Module):
    """
    U-Net++ architecture with ResNeXt-50 Encoder and Deep Supervision.

    Features:
    - Encoder: Pretrained se_resnext50_32x4d (ImageNet weights).
    - Decoder: Nested dense skip pathways (U-Net++).
    - Attention: SCSE modules in every decoder node.
    - Deep Supervision: Returns outputs from all nodes in the top decoder layer (L1-L4).
    """

    def __init__(self, n_classes=1, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        # 1. Encoder
        # Load pretrained ResNeXt-50
        # features_only=True returns feature maps at different scales
        # out_indices=(0, 1, 2, 3, 4) corresponds to strides 2, 4, 8, 16, 32
        self.encoder = timm.create_model(
            Config.ENCODER,
            features_only=True,
            pretrained=True if Config.ENCODER_WEIGHTS == "imagenet" else False,
            out_indices=(0, 1, 2, 3, 4),
            in_chans=Config.CHANNELS,
        )

        # Encoder Channels for se_resnext50_32x4d: [64, 256, 512, 1024, 2048]
        # x0_0 (s2), x1_0 (s4), x2_0 (s8), x3_0 (s16), x4_0 (s32)
        enc_ch = self.encoder.feature_info.channels()

        # Decoder Channels from Config: (256, 128, 64, 32, 16)
        # We map these to rows 4, 3, 2, 1, 0 respectively.
        # dec_ch[0] -> Row 0 (Target Resolution / 2)
        # ...
        # dec_ch[4] -> Row 4 (Bridge)
        # Reversing config tuple to match bottom-up indexing 0..4
        dec_ch = list(reversed(Config.DECODER_CHANNELS))  # [16, 32, 64, 128, 256]

        # 2. Decoder Nodes (Nested Skip Pathways)
        # Naming convention: conv{row}_{col}

        # Row 0 (Output stride 2)
        # Inputs: Previous node in row + Upsampled node from row below
        self.conv0_1 = ConvBlock(enc_ch[0] + enc_ch[1], dec_ch[0])
        self.conv0_2 = ConvBlock(enc_ch[0] + dec_ch[0] + dec_ch[1], dec_ch[0])
        self.conv0_3 = ConvBlock(enc_ch[0] + dec_ch[0] * 2 + dec_ch[1], dec_ch[0])
        self.conv0_4 = ConvBlock(enc_ch[0] + dec_ch[0] * 3 + dec_ch[1], dec_ch[0])

        # Row 1 (Output stride 4)
        self.conv1_1 = ConvBlock(enc_ch[1] + enc_ch[2], dec_ch[1])
        self.conv1_2 = ConvBlock(enc_ch[1] + dec_ch[1] + dec_ch[2], dec_ch[1])
        self.conv1_3 = ConvBlock(enc_ch[1] + dec_ch[1] * 2 + dec_ch[2], dec_ch[1])

        # Row 2 (Output stride 8)
        self.conv2_1 = ConvBlock(enc_ch[2] + enc_ch[3], dec_ch[2])
        self.conv2_2 = ConvBlock(enc_ch[2] + dec_ch[2] + dec_ch[3], dec_ch[2])

        # Row 3 (Output stride 16)
        self.conv3_1 = ConvBlock(enc_ch[3] + enc_ch[4], dec_ch[3])

        # 3. Final Output Layers (Deep Supervision)
        # All outputs are from Row 0 (stride 2), so they need 1 final upsample
        self.final_conv1 = nn.Conv2d(dec_ch[0], n_classes, 1)
        self.final_conv2 = nn.Conv2d(dec_ch[0], n_classes, 1)
        self.final_conv3 = nn.Conv2d(dec_ch[0], n_classes, 1)
        self.final_conv4 = nn.Conv2d(dec_ch[0], n_classes, 1)

    def forward(self, x):
        # 1. Encoder Pass
        features = self.encoder(x)
        x0_0 = features[0]  # Stride 2
        x1_0 = features[1]  # Stride 4
        x2_0 = features[2]  # Stride 8
        x3_0 = features[3]  # Stride 16
        x4_0 = features[4]  # Stride 32

        # Helper for upsampling
        def up(src, target_shape_tensor):
            return F.interpolate(
                src,
                size=target_shape_tensor.shape[2:],
                mode="bilinear",
                align_corners=True,
            )

        # 2. Decoder Pass (Nested U-Net)

        # Column 1
        # x3_1 input: x3_0 and up(x4_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, up(x4_0, x3_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, up(x3_0, x2_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, up(x2_0, x1_0)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, up(x1_0, x0_0)], 1))

        # Column 2
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, up(x3_1, x2_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, up(x2_1, x1_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, up(x1_1, x0_0)], 1))

        # Column 3
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, up(x2_2, x1_0)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, up(x1_2, x0_0)], 1))

        # Column 4 (Final Node)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, up(x1_3, x0_0)], 1))

        # 3. Output Generation
        # The decoder outputs are at stride 2 (64x64 for 128x128 input).
        # We need to upsample to original resolution.

        out4 = self.final_conv4(x0_4)
        out4 = F.interpolate(out4, scale_factor=2, mode="bilinear", align_corners=True)

        if self.training and self.deep_supervision:
            out1 = self.final_conv1(x0_1)
            out1 = F.interpolate(
                out1, scale_factor=2, mode="bilinear", align_corners=True
            )

            out2 = self.final_conv2(x0_2)
            out2 = F.interpolate(
                out2, scale_factor=2, mode="bilinear", align_corners=True
            )

            out3 = self.final_conv3(x0_3)
            out3 = F.interpolate(
                out3, scale_factor=2, mode="bilinear", align_corners=True
            )

            # Return list for Deep Supervision Loss
            return [out1, out2, out3, out4]

        # Inference / Validation returns only the final output
        return out4
