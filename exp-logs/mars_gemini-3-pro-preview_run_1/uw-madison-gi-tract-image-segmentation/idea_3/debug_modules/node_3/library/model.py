import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
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
        # Upsample x to match skip's spatial dimensions
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        if skip is not None:
            # Handle potential padding issues if shapes aren't perfectly divisible
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class UNet25D(nn.Module):
    def __init__(self, backbone_name=Config.BACKBONE, pretrained=True):
        super().__init__()

        # 1. Encoder (EfficientNet-B1)
        # features_only=True returns a list of feature maps
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # Get channel counts from the encoder
        # Typical EfficientNet-B1 feature channels: [16, 24, 40, 112, 320]
        # Strides: [2, 4, 8, 16, 32]
        encoder_channels = self.encoder.feature_info.channels()

        # 2. Decoder
        # We build the decoder from bottom (deepest) to top

        # Block 1: Input from s32 (320), Skip from s16 (112) -> Output 256
        self.decoder1 = DecoderBlock(
            in_channels=encoder_channels[4],
            skip_channels=encoder_channels[3],
            out_channels=256,
        )

        # Block 2: Input from prev (256), Skip from s8 (40) -> Output 128
        self.decoder2 = DecoderBlock(
            in_channels=256, skip_channels=encoder_channels[2], out_channels=128
        )

        # Block 3: Input from prev (128), Skip from s4 (24) -> Output 64
        self.decoder3 = DecoderBlock(
            in_channels=128, skip_channels=encoder_channels[1], out_channels=64
        )

        # Block 4: Input from prev (64), Skip from s2 (16) -> Output 32
        self.decoder4 = DecoderBlock(
            in_channels=64, skip_channels=encoder_channels[0], out_channels=32
        )

        # Block 5: Final Upsample to original resolution (s1).
        # No skip connection from encoder (stem is stride 2).
        self.decoder5 = DecoderBlock(in_channels=32, skip_channels=0, out_channels=16)

        # 3. Segmentation Head
        self.segmentation_head = nn.Conv2d(16, Config.NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        # x shape: (B, 3, H, W)

        # Encoder Pass
        features = self.encoder(x)
        # features[0]: stride 2
        # features[1]: stride 4
        # features[2]: stride 8
        # features[3]: stride 16
        # features[4]: stride 32

        # Decoder Pass
        x = self.decoder1(features[4], features[3])
        x = self.decoder2(x, features[2])
        x = self.decoder3(x, features[1])
        x = self.decoder4(x, features[0])

        # Final upsample to original size
        x = self.decoder5(x, skip=None)

        # Segmentation Head
        logits = self.segmentation_head(x)

        # Ensure output matches input spatial dimensions exactly
        # (Though decoder5 usually handles this, interpolation ensures safety)
        if logits.shape[-2:] != (Config.IMG_SIZE, Config.IMG_SIZE):
            logits = F.interpolate(
                logits,
                size=(Config.IMG_SIZE, Config.IMG_SIZE),
                mode="bilinear",
                align_corners=True,
            )

        return logits
