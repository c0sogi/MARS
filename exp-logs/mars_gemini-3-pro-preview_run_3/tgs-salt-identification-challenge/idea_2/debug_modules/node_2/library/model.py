import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from library.config import Config


class SCSEBlock(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation Block.
    Ref: https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels, reduction=16):
        super(SCSEBlock, self).__init__()
        # Channel Squeeze and Excitation
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        # Spatial Squeeze and Excitation
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # Concurrent combination: Additive
        return x * self.cSE(x) + x * self.sSE(x)


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block with SCSE Attention.
    Performs Upsampling -> Concatenation -> ConvBlock -> SCSE.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # The input to conv1 is the upsampled feature + the skip connection
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Attention Mechanism
        self.scse = SCSEBlock(out_channels)

    def forward(self, x, skip=None):
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Concatenate Skip Connection
        if skip is not None:
            # Handle slight shape mismatches due to padding/pooling
            if x.size() != skip.size():
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        # Convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Apply Attention
        x = self.scse(x)
        return x


class ResNeXtUNet(nn.Module):
    """
    U-Net with ResNeXt-50 (32x4d) Encoder and SCSE Attention Decoder.
    Designed for 4-channel input (RGB + Depth).
    """

    def __init__(self):
        super(ResNeXtUNet, self).__init__()

        # 1. Load Pretrained Encoder
        weights = "DEFAULT" if Config.ENCODER_WEIGHTS == "imagenet" else None
        encoder = torchvision.models.resnext50_32x4d(weights=weights)

        # 2. Modify First Layer for 4 Channels
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        old_conv = encoder.conv1
        new_conv = nn.Conv2d(
            Config.IN_CHANNELS,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias,
        )

        # Initialize weights
        with torch.no_grad():
            if weights is not None:
                # Copy RGB weights
                new_conv.weight[:, :3, :, :] = old_conv.weight
                # Initialize 4th channel (Depth) with mean of RGB weights
                new_conv.weight[:, 3:4, :, :] = torch.mean(
                    old_conv.weight, dim=1, keepdim=True
                )
            else:
                nn.init.kaiming_normal_(
                    new_conv.weight, mode="fan_out", nonlinearity="relu"
                )

        encoder.conv1 = new_conv

        # 3. Extract Encoder Layers for Skip Connections
        # Layer 0: Conv1 -> BN -> ReLU (64x64, 64ch)
        self.enc0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        # Layer 1: (32x32, 256ch)
        self.enc1 = encoder.layer1
        # Layer 2: (16x16, 512ch)
        self.enc2 = encoder.layer2
        # Layer 3: (8x8, 1024ch)
        self.enc3 = encoder.layer3
        # Layer 4: (4x4, 2048ch)
        self.enc4 = encoder.layer4

        # 4. Decoder Path
        # Center Block (Bottleneck)
        self.center = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Decoder Blocks
        # dec4: In(512) + Skip(1024) -> Out(256)
        self.dec4 = DecoderBlock(512, 1024, 256)
        # dec3: In(256) + Skip(512) -> Out(128)
        self.dec3 = DecoderBlock(256, 512, 128)
        # dec2: In(128) + Skip(256) -> Out(64)
        self.dec2 = DecoderBlock(128, 256, 64)
        # dec1: In(64) + Skip(64) -> Out(32)
        self.dec1 = DecoderBlock(64, 64, 32)

        # 5. Final Output Layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        # --- Encoder ---
        x0 = self.enc0(x)  # Stride 2 (64x64)
        x_pool = self.maxpool(x0)  # Stride 4 (32x32)
        x1 = self.enc1(x_pool)  # Stride 4 (32x32)
        x2 = self.enc2(x1)  # Stride 8 (16x16)
        x3 = self.enc3(x2)  # Stride 16 (8x8)
        x4 = self.enc4(x3)  # Stride 32 (4x4)

        # --- Center ---
        c = self.center(x4)

        # --- Decoder ---
        d4 = self.dec4(c, x3)  # -> 8x8
        d3 = self.dec3(d4, x2)  # -> 16x16
        d2 = self.dec2(d3, x1)  # -> 32x32
        d1 = self.dec1(d2, x0)  # -> 64x64

        # --- Final Upsample ---
        # Upsample from 64x64 to 128x128
        final = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=True)
        logits = self.final_conv(final)

        return logits
