import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
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
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        # Upsample x to match skip size
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ContrailUNet(nn.Module):
    """
    Standard U-Net with EfficientNet-B0 Encoder.
    Replaces U-Net++ to focus on optimization volume over architectural complexity.
    (Cite solution_lesson_node_00012)
    """

    def __init__(self):
        super().__init__()

        # --- Encoder ---
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # EfficientNet-B0 channels: [16, 24, 40, 112, 320]
        enc_channels = self.encoder.feature_info.channels()

        # --- Decoder ---
        # We start from the deepest feature (x4) and work up
        self.dec_channels = [256, 128, 64, 32, 16]

        # Block 1: Up(x4) + x3
        self.up1 = DecoderBlock(enc_channels[4], enc_channels[3], self.dec_channels[0])

        # Block 2: Up(d1) + x2
        self.up2 = DecoderBlock(
            self.dec_channels[0], enc_channels[2], self.dec_channels[1]
        )

        # Block 3: Up(d2) + x1
        self.up3 = DecoderBlock(
            self.dec_channels[1], enc_channels[1], self.dec_channels[2]
        )

        # Block 4: Up(d3) + x0
        self.up4 = DecoderBlock(
            self.dec_channels[2], enc_channels[0], self.dec_channels[3]
        )

        # Final Convolution
        # Output of up4 is same size as x0 (Stride 2)
        # We need one more upsample to get to Stride 1 (Original Image)
        # Since we don't have a skip connection from the stem (usually), we just upsample
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                self.dec_channels[3], self.dec_channels[4], kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(self.dec_channels[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dec_channels[4], 1, kernel_size=1),
        )

    def forward(self, x):
        # Encoder
        # x0: Stride 2, x1: Stride 4, x2: Stride 8, x3: Stride 16, x4: Stride 32
        features = self.encoder(x)
        x0, x1, x2, x3, x4 = features

        # Decoder
        d1 = self.up1(x4, x3)
        d2 = self.up2(d1, x2)
        d3 = self.up3(d2, x1)
        d4 = self.up4(d3, x0)

        # Final Upsample
        out = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=True)
        out = self.final_conv(out)

        return out
