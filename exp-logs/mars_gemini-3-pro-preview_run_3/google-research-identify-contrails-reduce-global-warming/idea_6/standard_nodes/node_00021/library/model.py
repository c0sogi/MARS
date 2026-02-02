import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concat (optional) -> Conv -> BN -> ReLU.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # We use bilinear interpolation for upsampling in the forward pass
        # The block input channels = in_channels (from previous decoder block) + skip_channels (from encoder)
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
            # Safety check for shape mismatch (e.g. due to odd input dimensions, though unlikely with 256x256)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        # Conv Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class ContrailUnet(nn.Module):
    """
    U-Net architecture with a ConvNeXt-Small backbone for Contrail Segmentation.
    """

    def __init__(self):
        super().__init__()

        # --- Encoder ---
        # Initialize ConvNeXt Small backbone
        # in_chans=6 adapts the first layer for the 6-channel input
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            in_chans=Config.N_CHANNELS,
        )

        # Retrieve channel counts for the feature maps
        # For ConvNeXt Small, typical channels are [96, 192, 384, 768]
        # Corresponding to strides [4, 8, 16, 32]
        encoder_channels = self.encoder.feature_info.channels()

        if len(encoder_channels) != 4:
            raise ValueError(
                f"Expected 4 feature levels from backbone, got {len(encoder_channels)}"
            )

        c1, c2, c3, c4 = encoder_channels

        # --- Decoder ---
        # Define decoder channel widths
        d_c1 = 256
        d_c2 = 128
        d_c3 = 64
        d_c4 = 32
        d_c5 = 16

        # Block 1: Input from bottleneck (c4, stride 32) -> Upsample to stride 16 -> Concat c3
        self.decoder1 = DecoderBlock(c4, c3, d_c1)

        # Block 2: Input d_c1 (stride 16) -> Upsample to stride 8 -> Concat c2
        self.decoder2 = DecoderBlock(d_c1, c2, d_c2)

        # Block 3: Input d_c2 (stride 8) -> Upsample to stride 4 -> Concat c1
        self.decoder3 = DecoderBlock(d_c2, c1, d_c3)

        # Block 4: Input d_c3 (stride 4) -> Upsample to stride 2
        # No skip connection available from encoder at stride 2 (stem is stride 4)
        self.decoder4 = DecoderBlock(d_c3, 0, d_c4)

        # Block 5: Input d_c4 (stride 2) -> Upsample to stride 1 (Original Resolution)
        self.decoder5 = DecoderBlock(d_c4, 0, d_c5)

        # --- Head ---
        # Final 1x1 Conv to produce binary logits
        self.segmentation_head = nn.Conv2d(d_c5, 1, kernel_size=1)

    def forward(self, x):
        # --- Encoder Pass ---
        # features = [f1, f2, f3, f4]
        # f1: stride 4
        # f2: stride 8
        # f3: stride 16
        # f4: stride 32
        features = self.encoder(x)
        f1, f2, f3, f4 = features

        # --- Decoder Pass ---
        x = self.decoder1(f4, f3)  # -> stride 16
        x = self.decoder2(x, f2)  # -> stride 8
        x = self.decoder3(x, f1)  # -> stride 4
        x = self.decoder4(x)  # -> stride 2
        x = self.decoder5(x)  # -> stride 1

        # --- Head ---
        logits = self.segmentation_head(x)

        return logits
