import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SegFormerEncoder(nn.Module):
    """
    SegFormer Encoder utilizing the MiT-B2 backbone.
    Extracts hierarchical features at strides 4, 8, 16, and 32.
    """

    def __init__(self, backbone_name=None, pretrained=None):
        super().__init__()
        if backbone_name is None:
            backbone_name = Config.BACKBONE
        if pretrained is None:
            pretrained = Config.PRETRAINED

        # features_only=True ensures the model returns a list of feature maps
        # from different stages rather than the final classification output.
        self.encoder = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True
        )

        # Retrieve channel information for the feature maps
        # For mit_b2, this is typically [64, 128, 320, 512]
        self.feature_info = self.encoder.feature_info.channels()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            List[torch.Tensor]: List of feature maps [1/4, 1/8, 1/16, 1/32].
        """
        features = self.encoder(x)
        return features


class DecoderBlock(nn.Module):
    """
    Standard U-Net style decoder block: Upsample -> Concat -> Conv -> BN -> ReLU.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # The input to conv1 is the concatenation of the upsampled features and the skip connection
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
        # Bilinear upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        if skip is not None:
            # Handle potential rounding errors in dimensions during interpolation
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=True
                )

            # Concatenate along channel dimension
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class UNetDecoder(nn.Module):
    """
    Custom U-Net Decoder adapted for SegFormer features.
    """

    def __init__(self, encoder_channels, decoder_channels=[256, 128, 64, 32]):
        super().__init__()
        # encoder_channels: [64, 128, 320, 512] for mit_b2
        # We process from deepest (lowest res) to shallowest (highest res).

        # Stage 1: Upsample 1/32 -> 1/16
        # Input: 512, Skip: 320 -> Output: 256
        self.block1 = DecoderBlock(
            encoder_channels[3], encoder_channels[2], decoder_channels[0]
        )

        # Stage 2: Upsample 1/16 -> 1/8
        # Input: 256, Skip: 128 -> Output: 128
        self.block2 = DecoderBlock(
            decoder_channels[0], encoder_channels[1], decoder_channels[1]
        )

        # Stage 3: Upsample 1/8 -> 1/4
        # Input: 128, Skip: 64 -> Output: 64
        self.block3 = DecoderBlock(
            decoder_channels[1], encoder_channels[0], decoder_channels[2]
        )

        # Stage 4: Final Upsample 1/4 -> 1/1
        # Since MiT stem reduces to 1/4 immediately, we don't have a 1/2 or 1/1 skip connection.
        # We perform a 4x upsample block.
        self.final_upsample = nn.Sequential(
            nn.Conv2d(
                decoder_channels[2],
                decoder_channels[3],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(
                decoder_channels[3],
                decoder_channels[3],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
        )

        # Final projection to class logits
        self.final_conv = nn.Conv2d(
            decoder_channels[3], Config.NUM_CLASSES, kernel_size=1
        )

    def forward(self, features):
        # features list order: [f1(1/4), f2(1/8), f3(1/16), f4(1/32)]
        f1, f2, f3, f4 = features

        x = self.block1(f4, f3)
        x = self.block2(x, f2)
        x = self.block3(x, f1)
        x = self.final_upsample(x)
        logits = self.final_conv(x)

        return logits


class HybridSegFormerUNet(nn.Module):
    """
    Hybrid Architecture: SegFormer (MiT-B2) Encoder + U-Net Decoder.
    Designed for high-precision ink detection.
    """

    def __init__(self):
        super().__init__()
        self.encoder = SegFormerEncoder()
        self.decoder = UNetDecoder(self.encoder.feature_info)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).
        Returns:
            torch.Tensor: Logits of shape (B, 1, H, W).
        """
        features = self.encoder(x)
        logits = self.decoder(features)
        return logits
