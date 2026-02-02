import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block:
    (Conv3x3 -> BN -> ReLU) x 2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block: Upsample -> Concat -> ConvBlock
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        # Concat
        if skip is not None:
            # Handle padding if shapes don't match exactly (e.g. odd dims)
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SegmentationModel(nn.Module):
    """
    Standard U-Net with EfficientNet Encoder.
    Replaces U-Net++ to reduce complexity and improve stability (Cite Lesson 13).
    """

    def __init__(self):
        super().__init__()

        # 1. Encoder
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # Get channel counts: [stride 2, 4, 8, 16, 32]
        enc_channels = self.encoder.feature_info.channels()

        # Decoder channels
        dec_channels = [256, 128, 64, 32]

        # 2. Decoder
        # Bottleneck (Stride 32) -> Stride 16
        self.center = ConvBlock(enc_channels[4], enc_channels[4])

        self.dec4 = DecoderBlock(enc_channels[4], enc_channels[3], dec_channels[0])
        self.dec3 = DecoderBlock(dec_channels[0], enc_channels[2], dec_channels[1])
        self.dec2 = DecoderBlock(dec_channels[1], enc_channels[1], dec_channels[2])
        self.dec1 = DecoderBlock(dec_channels[2], enc_channels[0], dec_channels[3])

        # 3. Final Head
        self.final_conv = nn.Conv2d(dec_channels[3], Config.NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[-2:]

        # Encoder
        features = self.encoder(x)
        # x0: stride 2, x1: stride 4, x2: stride 8, x3: stride 16, x4: stride 32
        x0, x1, x2, x3, x4 = features

        # Decoder
        center = self.center(x4)
        d4 = self.dec4(center, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        # Final Upsample from Stride 2 to Stride 1
        logits = self.final_conv(d1)
        logits = F.interpolate(
            logits, size=input_shape, mode="bilinear", align_corners=True
        )

        return logits
