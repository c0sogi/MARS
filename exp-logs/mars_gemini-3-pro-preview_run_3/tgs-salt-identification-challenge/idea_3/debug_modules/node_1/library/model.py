import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SCSEBlock(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE) Block.
    Recalibrates feature maps to suppress background noise and highlight salt regions.
    Reference: https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels, reduction=16):
        super(SCSEBlock, self).__init__()

        # Channel Excitation (cSE): Squeeze spatially, excite channel-wise
        # Handles "what" is present
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Spatial Excitation (sSE): Squeeze channels, excite spatially
        # Handles "where" it is present
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        # Concurrent: Additive combination of spatial and channel attention
        return x * self.cSE(x) + x * self.sSE(x)


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block consisting of:
    1. Bilinear Upsampling
    2. Concatenation with Encoder Skip Connection
    3. Convolutional Blocks (Conv-BN-ReLU)
    4. scSE Attention Block
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Bilinear upsampling is often smoother than ConvTranspose2d
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Convolution to fuse upsampled features and skip connection
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Refinement convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Attention Mechanism to refine the features before passing to next stage
        self.scse = SCSEBlock(out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            # Interpolate if dimensions don't match exactly (e.g. due to padding/pooling arithmetic)
            if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                x = F.interpolate(
                    x,
                    size=(skip.size(2), skip.size(3)),
                    mode="bilinear",
                    align_corners=True,
                )
            x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.scse(x)
        return x


class ResNeXtUNet(nn.Module):
    """
    U-Net architecture with ResNeXt-50 (32x4d) encoder and scSE attention in decoder.
    Input: (B, 3, H, W) - Image fused with Depth/Coord
    Output: (B, 1, H, W) - Logits
    """

    def __init__(self, n_classes=1, pretrained=True):
        super(ResNeXtUNet, self).__init__()

        # --------------------
        # Encoder (ResNeXt-50 32x4d)
        # --------------------
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
        self.encoder = models.resnext50_32x4d(weights=weights)

        # Extract layers for skip connections
        # Layer 0: Conv1 -> BN -> ReLU (Output: 64 ch, H/2)
        self.enc0 = nn.Sequential(
            self.encoder.conv1, self.encoder.bn1, self.encoder.relu
        )

        self.maxpool = self.encoder.maxpool  # Output: H/4

        self.enc1 = self.encoder.layer1  # Output: 256 ch, H/4
        self.enc2 = self.encoder.layer2  # Output: 512 ch, H/8
        self.enc3 = self.encoder.layer3  # Output: 1024 ch, H/16
        self.enc4 = self.encoder.layer4  # Output: 2048 ch, H/32 (Bottleneck)

        # --------------------
        # Decoder
        # --------------------
        # Dec4: Up from enc4 (2048), cat enc3 (1024) -> 256
        self.dec4 = DecoderBlock(in_channels=2048, skip_channels=1024, out_channels=256)

        # Dec3: Up from dec4 (256), cat enc2 (512) -> 128
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=512, out_channels=128)

        # Dec2: Up from dec3 (128), cat enc1 (256) -> 64
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=256, out_channels=64)

        # Dec1: Up from dec2 (64), cat enc0 (64) -> 32
        # Note: enc0 is H/2, enc1 input was H/4 (after maxpool)
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)

        # Final Upsample: Up from dec1 (32, H/2) -> Original Size (H)
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, kernel_size=1),
        )

    def forward(self, x):
        # --- Encoder Path ---
        x0 = self.enc0(x)  # 64, H/2
        x_pool = self.maxpool(x0)  # H/4
        x1 = self.enc1(x_pool)  # 256, H/4
        x2 = self.enc2(x1)  # 512, H/8
        x3 = self.enc3(x2)  # 1024, H/16
        x4 = self.enc4(x3)  # 2048, H/32

        # --- Decoder Path ---
        d4 = self.dec4(x4, x3)  # 256, H/16
        d3 = self.dec3(d4, x2)  # 128, H/8
        d2 = self.dec2(d3, x1)  # 64, H/4
        d1 = self.dec1(d2, x0)  # 32, H/2

        # --- Final Head ---
        out = self.final_up(d1)  # 32, H
        out = self.final_conv(out)  # n_classes, H

        return out
