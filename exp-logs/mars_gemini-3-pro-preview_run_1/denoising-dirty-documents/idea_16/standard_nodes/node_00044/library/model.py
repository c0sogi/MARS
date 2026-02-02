import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import MODEL_CONFIG, IMG_CHANNELS


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    Supports dilation and reflection padding.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dilation=1,
        use_reflection_padding=True,
    ):
        super().__init__()

        # Calculate padding to maintain spatial dimensions (assuming kernel_size=3)
        # padding = dilation * (kernel_size - 1) // 2
        padding = dilation

        padding_mode = "reflect" if use_reflection_padding else "zeros"

        # First convolution handles stride and dilation
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode=padding_mode,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolution keeps stride 1 and same dilation
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=padding,
            dilation=dilation,
            padding_mode=padding_mode,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    """
    Decoder Block: Upsample -> Concat -> ConvBlock
    Dynamically handles upsampling scale based on encoder resolution preservation.
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        upsample_scale=2,
        use_reflection_padding=True,
    ):
        super().__init__()
        self.upsample_scale = upsample_scale

        # Input channels to conv block is the sum of upsampled input and skip connection
        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels,
            stride=1,
            dilation=1,
            use_reflection_padding=use_reflection_padding,
        )

    def forward(self, x, skip):
        # Bilinear Upsampling
        if self.upsample_scale > 1:
            x = F.interpolate(
                x,
                scale_factor=self.upsample_scale,
                mode="bilinear",
                align_corners=False,
            )

        # Safety check for dimension alignment (e.g. odd padding scenarios)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)

        # Convolution
        x = self.conv(x)
        return x


class ResolutionPreservedUNet(nn.Module):
    """
    Resolution-Preserved Deep U-Net.

    Features:
    - 4-Level Encoder depth.
    - Modified Bottleneck (Stride 1, Dilation 2) to cap downsampling at 8x.
    - Bilinear Upsampling.
    - Sigmoid Activation.
    """

    def __init__(self):
        super().__init__()

        config = MODEL_CONFIG
        encoder_filters = config["encoder_filters"]
        decoder_filters = config["decoder_filters"]
        bottleneck_idx = config["bottleneck_block_index"]
        bottleneck_stride = config["bottleneck_stride"]
        bottleneck_dilation = config["bottleneck_dilation"]
        use_reflection = config["use_reflection_padding"]

        # --- Build Encoder ---
        self.encoder_blocks = nn.ModuleList()
        current_in_channels = IMG_CHANNELS

        # Track downsampling factor (scale) at each block to align decoder
        # Scale 1 = Original Resolution. Scale 2 = 1/2 Resolution, etc.
        self.scales = []
        current_scale = 1

        for i, out_channels in enumerate(encoder_filters):
            # Determine stride and dilation based on block index
            if i == bottleneck_idx:
                # Structural Innovation: Stride 1, Dilation 2
                stride = bottleneck_stride
                dilation = bottleneck_dilation
            elif i == 0:
                # First block usually preserves resolution
                stride = 1
                dilation = 1
            else:
                # Standard downsampling
                stride = 2
                dilation = 1

            block = ConvBlock(
                current_in_channels,
                out_channels,
                stride=stride,
                dilation=dilation,
                use_reflection_padding=use_reflection,
            )
            self.encoder_blocks.append(block)

            # Update scale tracking
            current_scale = current_scale * stride
            self.scales.append(current_scale)

            current_in_channels = out_channels

        # --- Build Decoder ---
        self.decoder_blocks = nn.ModuleList()

        # The input to the decoder is the output of the last encoder block (bottleneck)
        prev_channels = encoder_filters[-1]
        prev_scale = self.scales[-1]

        for i, out_channels in enumerate(decoder_filters):
            # Calculate which encoder block provides the skip connection
            # Decoder 0 connects to Encoder N-2 (since N-1 is the input)
            skip_idx = len(encoder_filters) - 2 - i

            skip_channels = encoder_filters[skip_idx]
            skip_scale = self.scales[skip_idx]

            # Determine required upsampling factor to match skip resolution
            upsample_scale = prev_scale // skip_scale
            # Ensure scale is at least 1
            upsample_scale = max(1, upsample_scale)

            dec_block = DecoderBlock(
                prev_channels,
                skip_channels,
                out_channels,
                upsample_scale=upsample_scale,
                use_reflection_padding=use_reflection,
            )
            self.decoder_blocks.append(dec_block)

            prev_channels = out_channels
            prev_scale = skip_scale  # After merging, we are at skip_scale resolution

        # --- Final Output ---
        self.final_conv = nn.Conv2d(prev_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder Pass
        skips = []
        for block in self.encoder_blocks:
            x = block(x)
            skips.append(x)

        # Decoder Pass
        # skips[-1] is the bottleneck output (input to decoder)
        x = skips[-1]

        for i, block in enumerate(self.decoder_blocks):
            # Retrieve corresponding skip connection
            skip_idx = len(skips) - 2 - i
            skip = skips[skip_idx]

            x = block(x, skip)

        # Final Projection
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x
