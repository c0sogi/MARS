import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel, SegformerConfig


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    U-Net style Decoder Block: Upsample -> Concat -> ConvBlock
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        # Bilinear upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Handle potential shape mismatch due to odd dimensions (though 512 is power of 2)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HybridSegFormer(nn.Module):
    """
    Precision-Enhanced Hybrid SegFormer (MiT-B2 + U-Net Decoder).

    Backbone: MiT-B2 (ImageNet weights)
    Decoder: U-Net style with skip connections
    """

    def __init__(self, pretrained=True):
        super(HybridSegFormer, self).__init__()

        # --- Backbone ---
        # MiT-B2 configuration
        model_name = "nvidia/mit-b2"
        if pretrained:
            self.encoder = SegformerModel.from_pretrained(model_name)
        else:
            config = SegformerConfig.from_pretrained(model_name)
            self.encoder = SegformerModel(config)

        # MiT-B2 Channels: [64, 128, 320, 512]
        # Strides: [4, 8, 16, 32]
        self.encoder_dims = [64, 128, 320, 512]

        # --- Decoder ---
        # Stage 4 (1/32, 512) -> Stage 3 (1/16, 320)
        self.dec1 = DecoderBlock(
            in_channels=self.encoder_dims[3],
            skip_channels=self.encoder_dims[2],
            out_channels=256,
        )

        # Stage 3 -> Stage 2 (1/8, 128)
        self.dec2 = DecoderBlock(
            in_channels=256, skip_channels=self.encoder_dims[1], out_channels=128
        )

        # Stage 2 -> Stage 1 (1/4, 64)
        self.dec3 = DecoderBlock(
            in_channels=128, skip_channels=self.encoder_dims[0], out_channels=64
        )

        # --- Head ---
        # Stage 1 (1/4, 64) -> Output (1/1, 1)
        self.final_conv = nn.Sequential(
            ConvBlock(64, 32), nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, 3, H, W)

        # Pass through encoder
        # output_hidden_states=True ensures we get the list of features
        outputs = self.encoder(x, output_hidden_states=True)
        features = outputs.hidden_states

        # Extract features from specific stages
        # MiT-B2 returns 4 feature maps corresponding to strides 4, 8, 16, 32
        s1 = features[0]  # 1/4, 64
        s2 = features[1]  # 1/8, 128
        s3 = features[2]  # 1/16, 320
        s4 = features[3]  # 1/32, 512

        # Decode
        x = self.dec1(s4, s3)  # -> 1/16
        x = self.dec2(x, s2)  # -> 1/8
        x = self.dec3(x, s1)  # -> 1/4

        # Final processing
        x = self.final_conv(x)  # -> 1/4, 1 channel

        # Final upsample to original resolution
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)

        return x
