import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block:
    Upsample -> Concat -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SegmentationModel(nn.Module):
    """
    Standard U-Net with EfficientNet Encoder.
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

        # Get channel counts from encoder feature maps
        # Indices 0..4 correspond to strides 2, 4, 8, 16, 32
        enc_channels = self.encoder.feature_info.channels()

        # Decoder channels (powers of 2)
        dec_channels = [256, 128, 64, 32, 16]

        # 2. Decoder Blocks
        # Block 1: Input from C4 (32), Skip C3 (16) -> Out 256
        self.dec1 = DecoderBlock(enc_channels[4], enc_channels[3], dec_channels[0])

        # Block 2: Input from Dec1, Skip C2 (8) -> Out 128
        self.dec2 = DecoderBlock(dec_channels[0], enc_channels[2], dec_channels[1])

        # Block 3: Input from Dec2, Skip C1 (4) -> Out 64
        self.dec3 = DecoderBlock(dec_channels[1], enc_channels[1], dec_channels[2])

        # Block 4: Input from Dec3, Skip C0 (2) -> Out 32
        self.dec4 = DecoderBlock(dec_channels[2], enc_channels[0], dec_channels[3])

        # 3. Final Upsample (Stride 2 -> Stride 1)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(
                dec_channels[3],
                dec_channels[4],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(dec_channels[4]),
            nn.ReLU(inplace=True),
        )

        # 4. Head
        self.head = nn.Conv2d(dec_channels[4], Config.NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[-2:]

        # Encoder
        features = self.encoder(x)  # [x2, x4, x8, x16, x32]

        # Decoder
        x = self.dec1(features[4], features[3])
        x = self.dec2(x, features[2])
        x = self.dec3(x, features[1])
        x = self.dec4(x, features[0])

        # Final Up
        x = self.final_up(x)

        # Head
        x = self.head(x)

        # Ensure exact output size matches input
        if x.shape[-2:] != input_shape:
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=True)

        return x
