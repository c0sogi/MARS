import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetPlusPlusBlock(nn.Module):
    """
    A single node in the U-Net++ decoder graph.
    It receives one upsampled feature map from the lower row,
    and 'n' skip connection feature maps from the same row (left side).
    """

    def __init__(self, in_channels, out_channels):
        super(UNetPlusPlusBlock, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels), ConvBlock(out_channels, out_channels)
        )

    def forward(self, x_down, *x_skips):
        """
        Args:
            x_down: Feature map from the layer below (needs upsampling).
            *x_skips: List of feature maps from the same layer (skip connections).
        """
        # Take the target size from the first skip connection
        # If no skips (shouldn't happen in U-Net++ logic except maybe top-left?), use x_down
        target_size = x_skips[0].shape[2:] if x_skips else x_down.shape[2:]

        # Upsample x_down to match target size
        if x_down.shape[2:] != target_size:
            x_down = F.interpolate(
                x_down, size=target_size, mode="bilinear", align_corners=True
            )

        # Concatenate all inputs
        x = torch.cat([*x_skips, x_down], dim=1)

        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """
    U-Net++ with ResNet34 Encoder (Output Stride 16).
    """

    def __init__(
        self,
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=Config.IN_CHANNELS,
        classes=Config.CLASSES,
        output_stride=Config.ENCODER_OUTPUT_STRIDE,
    ):
        super(UNetPlusPlus, self).__init__()

        # 1. Encoder (Backbone)
        # Using timm to load ResNet34 with dilated convolutions (output_stride=16)
        self.encoder = timm.create_model(
            encoder_name,
            features_only=True,
            output_stride=output_stride,
            pretrained=True if encoder_weights else False,
            in_chans=in_channels,
        )

        # Get channel counts from the encoder
        # Expected ResNet34 features:
        # idx 0: 64 ch (Stride 2)
        # idx 1: 64 ch (Stride 4)
        # idx 2: 128 ch (Stride 8)
        # idx 3: 256 ch (Stride 16)
        # idx 4: 512 ch (Stride 16 due to OS=16)
        dummy_input = torch.randn(1, in_channels, 256, 256)
        with torch.no_grad():
            features = self.encoder(dummy_input)

        enc_channels = [f.shape[1] for f in features]
        # We define the decoder channel counts to match the encoder levels
        # Row 0: matches enc_channels[0]
        # Row 1: matches enc_channels[1]
        # ...
        ch = enc_channels

        # 2. Decoder Nodes
        # The naming convention X_{i,j} where i is row (downsampling level), j is col (dense block index)

        # Column 1 (j=1)
        # Inputs: X_{i,0} (encoder), Up(X_{i+1, 0})
        self.conv_0_1 = UNetPlusPlusBlock(ch[0] + ch[1], ch[0])
        self.conv_1_1 = UNetPlusPlusBlock(ch[1] + ch[2], ch[1])
        self.conv_2_1 = UNetPlusPlusBlock(ch[2] + ch[3], ch[2])
        self.conv_3_1 = UNetPlusPlusBlock(ch[3] + ch[4], ch[3])

        # Column 2 (j=2)
        # Inputs: X_{i,0}, X_{i,1}, Up(X_{i+1, 1})
        # Concatenation size: ch[i] (X_i0) + ch[i] (X_i1) + ch[i+1] (Up_X_i+1_1)
        # Note: The output channels of conv_X_Y is ch[X]
        self.conv_0_2 = UNetPlusPlusBlock(ch[0] * 2 + ch[1], ch[0])
        self.conv_1_2 = UNetPlusPlusBlock(ch[1] * 2 + ch[2], ch[1])
        self.conv_2_2 = UNetPlusPlusBlock(ch[2] * 2 + ch[3], ch[2])

        # Column 3 (j=3)
        # Inputs: X_{i,0}, X_{i,1}, X_{i,2}, Up(X_{i+1, 2})
        self.conv_0_3 = UNetPlusPlusBlock(ch[0] * 3 + ch[1], ch[0])
        self.conv_1_3 = UNetPlusPlusBlock(ch[1] * 3 + ch[2], ch[1])

        # Column 4 (j=4) - Output Column
        # Inputs: X_{0,0}, X_{0,1}, X_{0,2}, X_{0,3}, Up(X_{1,3})
        self.conv_0_4 = UNetPlusPlusBlock(ch[0] * 4 + ch[1], ch[0])

        # 3. Final Segmentation Head
        self.final_conv = nn.Conv2d(ch[0], classes, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[2:]

        # Encoder
        features = self.encoder(x)
        x_0_0 = features[0]  # Stride 2
        x_1_0 = features[1]  # Stride 4
        x_2_0 = features[2]  # Stride 8
        x_3_0 = features[3]  # Stride 16
        x_4_0 = features[4]  # Stride 16 (Dilated)

        # Decoder Column 1
        x_0_1 = self.conv_0_1(x_1_0, x_0_0)
        x_1_1 = self.conv_1_1(x_2_0, x_1_0)
        x_2_1 = self.conv_2_1(x_3_0, x_2_0)
        x_3_1 = self.conv_3_1(x_4_0, x_3_0)

        # Decoder Column 2
        x_0_2 = self.conv_0_2(x_1_1, x_0_0, x_0_1)
        x_1_2 = self.conv_1_2(x_2_1, x_1_0, x_1_1)
        x_2_2 = self.conv_2_2(x_3_1, x_2_0, x_2_1)

        # Decoder Column 3
        x_0_3 = self.conv_0_3(x_1_2, x_0_0, x_0_1, x_0_2)
        x_1_3 = self.conv_1_3(x_2_2, x_1_0, x_1_1, x_1_2)

        # Decoder Column 4 (Final Node)
        x_0_4 = self.conv_0_4(x_1_3, x_0_0, x_0_1, x_0_2, x_0_3)

        # Final Head
        logits = self.final_conv(x_0_4)

        # Upsample to original input resolution (Stride 2 -> Stride 1)
        logits = F.interpolate(
            logits, size=input_shape, mode="bilinear", align_corners=True
        )

        return logits
