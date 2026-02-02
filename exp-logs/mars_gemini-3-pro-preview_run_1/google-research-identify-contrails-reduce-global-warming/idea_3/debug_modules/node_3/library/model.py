import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from library.config import Config


class ConvReluBN(nn.Module):
    """
    Basic Convolutional Block: Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvReluBN, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block: Upsample -> Concat -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # After concatenation, the channel count is in_channels + skip_channels
        self.conv1 = ConvReluBN(in_channels + skip_channels, out_channels)
        self.conv2 = ConvReluBN(out_channels, out_channels)

    def forward(self, x, skip):
        # Bilinear upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Concatenate with skip connection
        if skip is not None:
            # Ensure spatial dimensions match (robustness for odd sizes, though 256 is power of 2)
            if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                x = F.interpolate(
                    x,
                    size=(skip.size(2), skip.size(3)),
                    mode="bilinear",
                    align_corners=False,
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DilatedResNetUNet(nn.Module):
    """
    Dilated ResNet34 U-Net.

    Encoder: ResNet34.
    Decoder: Custom U-Net decoder.
    Note: ResNet34 does not support dilation in torchvision. We simulate the
    high-resolution feature map by upsampling layer4 output before fusion.
    """

    def __init__(self):
        super(DilatedResNetUNet, self).__init__()

        # ----------------------------------------------------------------------
        # Encoder (ResNet34)
        # ----------------------------------------------------------------------
        # Use ImageNet weights
        weights = (
            ResNet34_Weights.IMAGENET1K_V1
            if Config.ENCODER_WEIGHTS == "imagenet"
            else None
        )

        # Initialize ResNet34 without dilation (BasicBlock does not support it)
        self.encoder = resnet34(
            weights=weights,
            replace_stride_with_dilation=None,
        )

        # Modify input layer if necessary (ResNet default is 3 channels, which matches our Ash composite)
        if Config.IN_CHANNELS != 3:
            self.encoder.conv1 = nn.Conv2d(
                Config.IN_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # ----------------------------------------------------------------------
        # Decoder
        # ----------------------------------------------------------------------
        # Channel counts for ResNet34:
        # e0 (relu)   : 64
        # e1 (layer1) : 64
        # e2 (layer2) : 128
        # e3 (layer3) : 256
        # e4 (layer4) : 512

        # Center Block: Fuses e4 (512) and e3 (256). Both are 16x16.
        # No upsampling here, just concatenation and processing.
        self.center_conv1 = ConvReluBN(512 + 256, 256)
        self.center_conv2 = ConvReluBN(256, 256)

        # Decoder Stage 3: Upsample Center (256) -> Concat e2 (128) -> Output 128
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)

        # Decoder Stage 2: Upsample Dec3 (128) -> Concat e1 (64) -> Output 64
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)

        # Decoder Stage 1: Upsample Dec2 (64) -> Concat e0 (64) -> Output 64
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        # ----------------------------------------------------------------------
        # Segmentation Head
        # ----------------------------------------------------------------------
        # Final upsample from 128x128 (Dec1 output) to 256x256
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvReluBN(64, 32),
            nn.Conv2d(32, Config.CLASSES, kernel_size=1),
        )

    def forward(self, x):
        # ----------------------------------------------------------------------
        # Encoder Pass
        # ----------------------------------------------------------------------
        # Input: (B, 3, 256, 256)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        e0 = self.encoder.relu(x)  # (B, 64, 128, 128)

        x = self.encoder.maxpool(e0)
        e1 = self.encoder.layer1(x)  # (B, 64, 64, 64)
        e2 = self.encoder.layer2(e1)  # (B, 128, 32, 32)
        e3 = self.encoder.layer3(e2)  # (B, 256, 16, 16)
        e4 = self.encoder.layer4(e3)  # (B, 512, 8, 8)

        # ----------------------------------------------------------------------
        # Decoder Pass
        # ----------------------------------------------------------------------
        # Center: Fuse e4 and e3
        # e4 is 8x8, e3 is 16x16. Upsample e4 to 16x16.
        e4_up = F.interpolate(
            e4, size=e3.shape[2:], mode="bilinear", align_corners=False
        )

        center = torch.cat([e4_up, e3], dim=1)
        center = self.center_conv1(center)
        center = self.center_conv2(center)  # (B, 256, 16, 16)

        # Decode
        d3 = self.dec3(center, e2)  # -> (B, 128, 32, 32)
        d2 = self.dec2(d3, e1)  # -> (B, 64, 64, 64)
        d1 = self.dec1(d2, e0)  # -> (B, 64, 128, 128)

        # ----------------------------------------------------------------------
        # Head
        # ----------------------------------------------------------------------
        logits = self.final_conv(d1)  # -> (B, 1, 256, 256)

        return logits
