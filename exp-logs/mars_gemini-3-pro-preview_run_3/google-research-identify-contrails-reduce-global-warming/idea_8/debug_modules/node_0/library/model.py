import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concat -> ConvBlock
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # The input to conv1 is the concatenation of the upsampled feature and the skip connection
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

        # Concatenate with skip connection if provided
        if skip is not None:
            # Ensure dimensions match (handle potential rounding errors in odd dimensions)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ContrailUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ==============================
        # 1. Encoder (Backbone)
        # ==============================
        # Load ConvNeXt-Base with ImageNet weights.
        # features_only=True allows us to get intermediate feature maps for the U-Net.
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # ==============================
        # 2. Input Adaptation (Stem)
        # ==============================
        # The default stem accepts 3 channels (RGB). We need to modify it for 9 channels.
        # ConvNeXt stem is typically a Sequential containing a Conv2d at index 0.
        original_stem_conv = self.encoder.stem[0]

        # Create a new Conv2d layer with the target input channels
        new_stem_conv = nn.Conv2d(
            in_channels=Config.INPUT_CHANNELS,
            out_channels=original_stem_conv.out_channels,
            kernel_size=original_stem_conv.kernel_size,
            stride=original_stem_conv.stride,
            padding=original_stem_conv.padding,
            bias=original_stem_conv.bias is not None,
        )

        # Weight Initialization Strategy:
        # To benefit from pretraining, we copy the weights from the original 3 channels
        # and repeat them for the new channels. We divide by 3 to keep the initial
        # activation magnitude roughly the same as the original model.
        with torch.no_grad():
            # original weight shape: (Out, 3, K, K)
            # new weight shape: (Out, 9, K, K)
            repeat_factor = Config.INPUT_CHANNELS // 3
            new_stem_conv.weight[:] = original_stem_conv.weight.repeat(
                1, repeat_factor, 1, 1
            ) / float(repeat_factor)

            if original_stem_conv.bias is not None:
                new_stem_conv.bias[:] = original_stem_conv.bias

        # Replace the layer in the encoder
        self.encoder.stem[0] = new_stem_conv

        # ==============================
        # 3. Decoder Construction
        # ==============================
        # Get channel counts for the feature maps
        # Typically for ConvNeXt-Base: [128, 256, 512, 1024]
        # Corresponding to strides: [4, 8, 16, 32]
        enc_channels = self.encoder.feature_info.channels()
        c0, c1, c2, c3 = enc_channels

        # Decoder Block 1: Input from Stage 3 (Stride 32), Skip from Stage 2 (Stride 16)
        self.decoder1 = DecoderBlock(in_channels=c3, skip_channels=c2, out_channels=256)

        # Decoder Block 2: Input from Decoder 1, Skip from Stage 1 (Stride 8)
        self.decoder2 = DecoderBlock(
            in_channels=256, skip_channels=c1, out_channels=128
        )

        # Decoder Block 3: Input from Decoder 2, Skip from Stage 0 (Stride 4)
        self.decoder3 = DecoderBlock(in_channels=128, skip_channels=c0, out_channels=64)

        # ==============================
        # 4. Segmentation Head
        # ==============================
        # The output of decoder3 is at Stride 4. We need to upsample 4x to match input resolution.
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, Config.NUM_CLASSES, kernel_size=1),
        )

    def forward(self, x):
        # Encoder Pass
        # features = [c0, c1, c2, c3]
        features = self.encoder(x)
        s0, s1, s2, s3 = features

        # Decoder Pass
        d1 = self.decoder1(s3, s2)  # Upsample s3 (32x) -> join s2 (16x) -> out (16x)
        d2 = self.decoder2(d1, s1)  # Upsample d1 (16x) -> join s1 (8x)  -> out (8x)
        d3 = self.decoder3(d2, s0)  # Upsample d2 (8x)  -> join s0 (4x)  -> out (4x)

        # Final Head
        logits = self.final_up(d3)  # Upsample d3 (4x)  -> out (1x)

        return logits
