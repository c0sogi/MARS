import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concat Skip -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """

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
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Concatenate skip connection
        if skip is not None:
            # Handle potential shape mismatch due to rounding in odd dimensions
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


class ContrailUNet(nn.Module):
    """
    U-Net architecture with ConvNeXt-Small backbone.
    Modified to accept 9-channel input (Multi-Order Temporal Composites).
    """

    def __init__(self):
        super().__init__()

        # 1. Encoder (Backbone)
        # Load pretrained ConvNeXt-Small, extract features at different stages
        # out_indices correspond to strides: 0->4, 1->8, 2->16, 3->32
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # 2. Modify Input Layer (Stem)
        # Original stem[0] is a Conv2d(3, 96, kernel_size=4, stride=4)
        original_stem = self.encoder.stem[0]

        # Create new Conv2d with 9 input channels
        new_stem = nn.Conv2d(
            in_channels=Config.INPUT_CHANNELS,
            out_channels=original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            bias=original_stem.bias is not None,
        )

        # Initialize weights: Copy RGB weights to first 3 channels to preserve spatial features
        # The remaining 6 channels (temporal differences) are initialized randomly
        with torch.no_grad():
            new_stem.weight[:, :3, :, :] = original_stem.weight
            if original_stem.bias is not None:
                new_stem.bias = original_stem.bias

        # Replace the layer in the encoder
        self.encoder.stem[0] = new_stem

        # Get channel counts for skip connections
        # ConvNeXt-Small usually: [96, 192, 384, 768]
        feature_channels = self.encoder.feature_info.channels()
        c0, c1, c2, c3 = feature_channels

        # 3. Decoder
        # Bottleneck (Stride 32) -> Up1 (Stride 16)
        self.decoder4 = DecoderBlock(c3, c2, 256)

        # Up1 (Stride 16) -> Up2 (Stride 8)
        self.decoder3 = DecoderBlock(256, c1, 128)

        # Up2 (Stride 8) -> Up3 (Stride 4)
        self.decoder2 = DecoderBlock(128, c0, 64)

        # Up3 (Stride 4) -> Up4 (Stride 2)
        # No skip connection here as encoder starts at stride 4
        self.decoder1 = DecoderBlock(64, 0, 32)

        # 4. Final Segmentation Head
        # Up4 (Stride 2) -> Original Resolution (Stride 1)
        self.final_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),  # Logits
        )

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        # features list: [stride4, stride8, stride16, stride32]
        f0, f1, f2, f3 = features

        # Decoder
        x = self.decoder4(f3, f2)  # 1/32 -> 1/16
        x = self.decoder3(x, f1)  # 1/16 -> 1/8
        x = self.decoder2(x, f0)  # 1/8 -> 1/4
        x = self.decoder1(x)  # 1/4 -> 1/2

        # Head
        logits = self.final_head(x)  # 1/2 -> 1/1

        return logits
