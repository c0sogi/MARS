import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concat -> Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # We use bilinear interpolation for upsampling in forward(), so no layer needed here
        # The input to conv1 will be the concatenation of upsampled input and skip connection
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
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Concatenate skip connection
        if skip is not None:
            # Ensure spatial dimensions match (handle minor rounding errors in padding)
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


class ContrailUNet(nn.Module):
    """
    U-Net architecture with an EfficientNet backbone.
    Designed to accept 6-channel inputs (Ash composite + Temporal Difference).
    """

    def __init__(
        self,
        encoder_name="efficientnet_b0",
        encoder_weights="imagenet",
        in_channels=6,
        classes=1,
    ):
        super().__init__()

        # --- Encoder ---
        # Load EfficientNet with timm
        # in_chans=6 will automatically adapt the first layer weights (repeating RGB weights)
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=(encoder_weights == "imagenet"),
            features_only=True,
            in_chans=in_channels,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Get the number of channels for each feature map
        # Expected strides for EfficientNet-B0: [2, 4, 8, 16, 32]
        feature_channels = self.encoder.feature_info.channels()

        # --- Decoder ---
        # We define decoder channel widths (hyperparameters)
        dec_channels = [256, 128, 64, 32, 16]

        # Decoder Stage 1: Input from Encoder Stage 4 (Stride 32), Skip from Stage 3 (Stride 16)
        self.decoder1 = DecoderBlock(
            feature_channels[4], feature_channels[3], dec_channels[0]
        )

        # Decoder Stage 2: Input from Dec1 (Stride 16), Skip from Stage 2 (Stride 8)
        self.decoder2 = DecoderBlock(
            dec_channels[0], feature_channels[2], dec_channels[1]
        )

        # Decoder Stage 3: Input from Dec2 (Stride 8), Skip from Stage 1 (Stride 4)
        self.decoder3 = DecoderBlock(
            dec_channels[1], feature_channels[1], dec_channels[2]
        )

        # Decoder Stage 4: Input from Dec3 (Stride 4), Skip from Stage 0 (Stride 2)
        self.decoder4 = DecoderBlock(
            dec_channels[2], feature_channels[0], dec_channels[3]
        )

        # Decoder Stage 5: Input from Dec4 (Stride 2), No Skip (Upsample to Stride 1)
        self.decoder5 = DecoderBlock(dec_channels[3], 0, dec_channels[4])

        # --- Head ---
        self.segmentation_head = nn.Conv2d(dec_channels[4], classes, kernel_size=1)

    def forward(self, x):
        # Encoder Pass
        features = self.encoder(x)
        # features list contains maps at strides: [2, 4, 8, 16, 32]
        f0, f1, f2, f3, f4 = features

        # Decoder Pass
        x_dec = self.decoder1(f4, f3)  # 32 -> 16
        x_dec = self.decoder2(x_dec, f2)  # 16 -> 8
        x_dec = self.decoder3(x_dec, f1)  # 8 -> 4
        x_dec = self.decoder4(x_dec, f0)  # 4 -> 2
        x_dec = self.decoder5(x_dec)  # 2 -> 1

        # Output Head
        logits = self.segmentation_head(x_dec)

        # Final safety check for interpolation to input size (if input wasn't divisible by 32)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits, size=x.shape[-2:], mode="bilinear", align_corners=True
            )

        return logits
