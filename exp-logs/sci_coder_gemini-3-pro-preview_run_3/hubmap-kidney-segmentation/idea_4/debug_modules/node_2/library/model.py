import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config
from library.stain_deconv import StainDeconvolution


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        if skip is not None:
            # Handle padding if shapes don't match exactly (though they should with padding=same)
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class StainNet(nn.Module):
    """
    Stain-Deconvolved U-Net with ConvNeXt-Tiny backbone.
    Re-implemented using timm and custom decoder to avoid dependency on segmentation_models_pytorch.
    """

    def __init__(self):
        super(StainNet, self).__init__()

        # 1. Stain Deconvolution Layer
        self.stain_deconv = StainDeconvolution()

        # 2. Encoder (Backbone) using timm
        # features_only=True returns a list of feature maps
        # in_chans=5 allows us to pass the 5-channel input directly
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            in_chans=Config.INPUT_CHANNELS,
        )

        # Get channel counts from the backbone
        # ConvNeXt-Tiny typically returns features with strides [4, 8, 16, 32]
        feature_channels = self.encoder.feature_info.channels()

        # 3. Decoder
        # We build a U-Net decoder.
        # Encoder features: [C1 (s4), C2 (s8), C3 (s16), C4 (s32)]

        # Bottleneck is C4.
        # Block 1: Upsample C4 -> concat C3
        self.decoder4 = DecoderBlock(feature_channels[3], feature_channels[2], 256)

        # Block 2: Upsample -> concat C2
        self.decoder3 = DecoderBlock(256, feature_channels[1], 128)

        # Block 3: Upsample -> concat C1
        self.decoder2 = DecoderBlock(128, feature_channels[0], 64)

        # Block 4: Upsample -> Output resolution (Stride 4 to 1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 4. Segmentation Head
        self.head = nn.Conv2d(32, Config.NUM_CLASSES, kernel_size=1)

        # Deep Supervision Heads
        if Config.DEEP_SUPERVISION:
            self.head_d3 = nn.Conv2d(128, Config.NUM_CLASSES, kernel_size=1)
            self.head_d4 = nn.Conv2d(256, Config.NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input RGB image batch of shape (B, 3, H, W).
        """
        input_shape = x.shape[-2:]

        # 1. Stain Deconvolution: (B, 3, H, W) -> (B, 5, H, W)
        x_aug = self.stain_deconv(x)

        # 2. Encoder
        features = self.encoder(x_aug)
        # features = [c1, c2, c3, c4]
        c1, c2, c3, c4 = features

        # 3. Decoder
        d4 = self.decoder4(c4, c3)  # Stride 32 -> 16
        d3 = self.decoder3(d4, c2)  # Stride 16 -> 8
        d2 = self.decoder2(d3, c1)  # Stride 8 -> 4

        # Final processing at stride 4
        x_out = self.final_conv(d2)

        # 4. Head & Upsample to original size
        logits = self.head(x_out)
        logits = F.interpolate(
            logits, size=input_shape, mode="bilinear", align_corners=True
        )

        if Config.DEEP_SUPERVISION:
            # Aux Head 1 (d3)
            logits_d3 = self.head_d3(d3)
            logits_d3 = F.interpolate(
                logits_d3, size=input_shape, mode="bilinear", align_corners=True
            )

            # Aux Head 2 (d4)
            logits_d4 = self.head_d4(d4)
            logits_d4 = F.interpolate(
                logits_d4, size=input_shape, mode="bilinear", align_corners=True
            )

            return [logits, logits_d3, logits_d4]

        return logits
