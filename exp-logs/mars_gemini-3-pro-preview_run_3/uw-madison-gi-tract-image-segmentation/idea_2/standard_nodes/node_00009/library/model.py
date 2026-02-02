import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DecoderBlock(nn.Module):
    """
    A standard U-Net decoder block that upsamples the input, concatenates with a skip connection,
    and applies convolutions.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # The input channels to conv1 are the sum of the upsampled input and the skip connection
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

    def forward(self, x, skip=None):
        # Upsample the input from the deeper layer
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # If a skip connection is provided, concatenate it
        if skip is not None:
            # Handle slight shape mismatches due to padding/rounding in the encoder
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class UNetEfficientNet(nn.Module):
    """
    U-Net architecture with an EfficientNet-B1 backbone.
    """

    def __init__(
        self, backbone_name=Config.BACKBONE, pretrained=True, classes=Config.NUM_CLASSES
    ):
        super(UNetEfficientNet, self).__init__()

        # Encoder: EfficientNet-B1
        # features_only=True returns a list of feature maps at different strides
        # in_chans=3 allows for 2.5D input (slices t-1, t, t+1)
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # Retrieve channel counts for the feature maps
        # Indices typically correspond to strides: 0 (s2), 1 (s4), 2 (s8), 3 (s16), 4 (s32)
        encoder_channels = self.encoder.feature_info.channels()

        # Decoder Setup
        # We build the decoder path from the bottom (deepest) up

        # Decoder Block 4: Takes stride 32 features, fuses with stride 16 skip
        self.decoder4 = DecoderBlock(
            in_channels=encoder_channels[4],
            skip_channels=encoder_channels[3],
            out_channels=256,
        )

        # Decoder Block 3: Takes stride 16 output, fuses with stride 8 skip
        self.decoder3 = DecoderBlock(
            in_channels=256, skip_channels=encoder_channels[2], out_channels=128
        )

        # Decoder Block 2: Takes stride 8 output, fuses with stride 4 skip
        self.decoder2 = DecoderBlock(
            in_channels=128, skip_channels=encoder_channels[1], out_channels=64
        )

        # Decoder Block 1: Takes stride 4 output, fuses with stride 2 skip
        self.decoder1 = DecoderBlock(
            in_channels=64, skip_channels=encoder_channels[0], out_channels=32
        )

        # Final Upsampling Block: Takes stride 2 output, upsamples to stride 1 (original size)
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Segmentation Head: Maps to number of classes
        self.segmentation_head = nn.Conv2d(16, classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        features = self.encoder(x)
        # features is a list of tensors: [f0(s2), f1(s4), f2(s8), f3(s16), f4(s32)]
        f0, f1, f2, f3, f4 = features

        # --- Decoder ---
        # Pass deep features up and concatenate with shallower features
        x = self.decoder4(f4, f3)
        x = self.decoder3(x, f2)
        x = self.decoder2(x, f1)
        x = self.decoder1(x, f0)

        # --- Final Head ---
        x = self.final_upsample(x)
        logits = self.segmentation_head(x)

        # Ensure output matches input spatial dimensions exactly
        # (Handles cases where input dimensions might not be perfectly divisible by 32)
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(
                logits, size=x.shape[2:], mode="bilinear", align_corners=True
            )

        return logits
