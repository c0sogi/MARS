import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DoubleConv(nn.Module):
    """
    A standard double convolution block: (Conv -> BN -> ReLU) * 2.
    Used within the decoder to process features after concatenation.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DecoderBlock(nn.Module):
    """
    A single decoder stage: Upsample -> Concat with Skip -> DoubleConv.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Bilinear upsampling by factor of 2
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # The input to DoubleConv will be the concatenation of the upsampled features
        # and the skip connection features.
        self.conv = DoubleConv(in_channels + skip_channels, out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle potential slight mismatches in dimensions (though unlikely with 256x256)
        if x.shape != skip.shape:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        # Concatenate along channel dimension
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ConvNeXtUNet(nn.Module):
    """
    U-Net architecture with a ConvNeXt Tiny backbone.
    Designed for 6-channel input (Ash composite + Temporal Difference).
    """

    def __init__(self):
        super().__init__()

        # 1. Encoder (Backbone)
        # Load pre-trained ConvNeXt Tiny.
        # features_only=True returns the feature maps from intermediate stages.
        # in_chans=Config.N_CHANNELS (6) adapts the first layer.
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            in_chans=Config.N_CHANNELS,
        )

        # Get channel counts for the feature maps.
        # ConvNeXt Tiny typically returns features at strides [4, 8, 16, 32].
        # Channels are usually [96, 192, 384, 768].
        feature_channels = self.encoder.feature_info.channels()
        c0, c1, c2, c3 = feature_channels

        # 2. Decoder Stages
        # We start from the deepest feature map (c3, stride 32) and move up.

        # Block 1: Upsample c3 (768) + Skip c2 (384) -> Output 384
        self.up1 = DecoderBlock(c3, c2, 384)

        # Block 2: Upsample result (384) + Skip c1 (192) -> Output 192
        self.up2 = DecoderBlock(384, c1, 192)

        # Block 3: Upsample result (192) + Skip c0 (96) -> Output 96
        # At this point, we are at stride 4 (64x64 resolution).
        self.up3 = DecoderBlock(192, c0, 96)

        # 3. Final Upsampling (Stride 4 -> Stride 1)
        # Since the backbone stem downsamples by 4, we need two more upsampling steps
        # to reach the original image resolution (256x256).

        # 64x64 -> 128x128
        self.final_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(96, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # 128x128 -> 256x256
        self.final_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(48, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

        # 4. Segmentation Head
        # Projects final features to 1 channel (logits)
        self.out_conv = nn.Conv2d(24, 1, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.
        x: (B, 6, H, W)
        Returns: (B, 1, H, W) logits
        """
        # Encoder
        features = self.encoder(x)
        # f0: stride 4, f1: stride 8, f2: stride 16, f3: stride 32
        f0, f1, f2, f3 = features

        # Decoder
        x = self.up1(f3, f2)  # Stride 32 -> 16
        x = self.up2(x, f1)  # Stride 16 -> 8
        x = self.up3(x, f0)  # Stride 8 -> 4

        # Restore resolution
        x = self.final_up1(x)  # Stride 4 -> 2
        x = self.final_up2(x)  # Stride 2 -> 1

        # Prediction
        logits = self.out_conv(x)

        return logits
