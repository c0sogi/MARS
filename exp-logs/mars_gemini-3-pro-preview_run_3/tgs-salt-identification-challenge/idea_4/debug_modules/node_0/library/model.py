import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

# --- Building Blocks ---


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE).
    Ref: https://arxiv.org/abs/1803.02539
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ConvBnRelu(nn.Module):
    """
    Standard Convolution -> BatchNorm -> ReLU block.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    """
    U-Net decoder block with optional scSE attention.
    """

    def __init__(self, in_channels, out_channels, use_scse=True):
        super().__init__()
        self.conv1 = ConvBnRelu(in_channels, out_channels)
        self.conv2 = ConvBnRelu(out_channels, out_channels)
        self.scse = SCSEModule(out_channels) if use_scse else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.scse(x)
        return x


# --- Main Model ---


class SaltModel(nn.Module):
    """
    U-Net++ with ResNeXt-50 Encoder and scSE Attention.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Encoder Setup
        # Load ResNeXt-50 (32x4d).
        # Feature channels: [64, 256, 512, 1024, 2048]
        # Strides: [2, 4, 8, 16, 32]
        self.encoder = timm.create_model(
            config.ENCODER_NAME,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Modify first layer to accept 4 channels (RGB + Depth)
        original_conv = self.encoder.conv1
        new_conv = nn.Conv2d(
            config.IN_CHANNELS,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # Initialize the new layer
        # Copy weights for RGB channels, initialize Depth channel with mean of RGB
        with torch.no_grad():
            new_conv.weight[:, :3] = original_conv.weight
            new_conv.weight[:, 3] = original_conv.weight.mean(dim=1)

        self.encoder.conv1 = new_conv

        # Channel Definitions
        enc_ch = [64, 256, 512, 1024, 2048]
        dec_ch = [64, 128, 256, 512, 1024]  # Decoder width

        # 2. Decoder Construction (U-Net++ Nested Skip Connections)
        # Notation: conv{row}_{col} where row is scale level, col is dense depth

        # Column 1 (Standard U-Net connections)
        # Input: Encoder feature + Upsampled lower feature
        self.conv0_1 = DecoderBlock(enc_ch[0] + dec_ch[1], dec_ch[0])
        self.conv1_1 = DecoderBlock(enc_ch[1] + dec_ch[2], dec_ch[1])
        self.conv2_1 = DecoderBlock(enc_ch[2] + dec_ch[3], dec_ch[2])
        self.conv3_1 = DecoderBlock(enc_ch[3] + dec_ch[4], dec_ch[3])

        # Column 2 (Nested)
        # Input: Encoder feature + Col 1 feature + Upsampled lower feature
        self.conv0_2 = DecoderBlock(enc_ch[0] + dec_ch[0] + dec_ch[1], dec_ch[0])
        self.conv1_2 = DecoderBlock(enc_ch[1] + dec_ch[1] + dec_ch[2], dec_ch[1])
        self.conv2_2 = DecoderBlock(enc_ch[2] + dec_ch[2] + dec_ch[3], dec_ch[2])

        # Column 3 (Nested)
        # Input: Encoder + Col 1 + Col 2 + Upsampled lower
        self.conv0_3 = DecoderBlock(enc_ch[0] + dec_ch[0] * 2 + dec_ch[1], dec_ch[0])
        self.conv1_3 = DecoderBlock(enc_ch[1] + dec_ch[1] * 2 + dec_ch[2], dec_ch[1])

        # Column 4 (Nested - Final Layer)
        # Input: Encoder + Col 1 + Col 2 + Col 3 + Upsampled lower
        self.conv0_4 = DecoderBlock(enc_ch[0] + dec_ch[0] * 3 + dec_ch[1], dec_ch[0])

        # 3. Final Segmentation Head
        # The output of conv0_4 is at stride 2 (64x64 for 128x128 input).
        # We need to upsample to full resolution.
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(dec_ch[0], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        # Encoder Pass
        # x: (B, 4, 128, 128)
        feats = self.encoder(x)
        # x0: (B, 64, 64, 64)   (Stride 2)
        # x1: (B, 256, 32, 32)  (Stride 4)
        # x2: (B, 512, 16, 16)  (Stride 8)
        # x3: (B, 1024, 8, 8)   (Stride 16)
        # x4: (B, 2048, 4, 4)   (Stride 32)
        x0_0, x1_0, x2_0, x3_0, x4_0 = feats

        # Decoder Column 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0, x3_0)], 1))

        # Decoder Column 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], 1))

        # Decoder Column 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], 1))

        # Decoder Column 4
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], 1)
        )

        # Final Head
        logits = self.final_up(x0_4)

        return logits

    def _up(self, x, target):
        """Helper to upsample x to match target spatial size."""
        if x.shape[2:] != target.shape[2:]:
            return F.interpolate(
                x, size=target.shape[2:], mode="bilinear", align_corners=True
            )
        return x
