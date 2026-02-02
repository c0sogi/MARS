import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    """
    Decoder Block: Upsample -> Concat with Skip -> ConvBlock
    """

    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_c + skip_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)

        # Ensure dimensions match (handles potential rounding issues, though 256x256 is safe)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MobileNetUNet(nn.Module):
    """
    U-Net with MobileNetV3-Small Encoder for Contrail Segmentation.

    Input: 6 Channels (Ash t=4, Difference t=4-t=3)
    Output: 1 Channel (Logits)
    """

    def __init__(self, in_channels=Config.IN_CHANNELS, num_classes=1):
        super().__init__()

        # Load Pretrained Backbone
        # MobileNetV3-Small is lightweight and efficient
        backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        features = backbone.features

        # ----------------------------------------------------------------
        # 1. Modify Input Layer
        # ----------------------------------------------------------------
        # Original first layer: Conv2d(3, 16, stride=2)
        # We need Conv2d(6, 16, stride=2)
        original_layer = features[0][0]

        new_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_layer.out_channels,
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride,
            padding=original_layer.padding,
            bias=original_layer.bias is not None,
        )

        # Initialize weights for the new layer
        # Copy weights for the first 3 channels (RGB/Ash)
        # Initialize the next 3 channels (Difference) with the same weights
        # to provide a stable initialization for the motion features.
        with torch.no_grad():
            new_layer.weight[:, :3] = original_layer.weight
            if in_channels > 3:
                new_layer.weight[:, 3:6] = original_layer.weight

        features[0][0] = new_layer

        # ----------------------------------------------------------------
        # 2. Define Encoder Stages
        # ----------------------------------------------------------------
        # Extract feature maps at different scales based on MobileNetV3 architecture
        # Indices determined by analyzing stride locations in mobilenet_v3_small

        self.enc0 = features[0]  # Stride 2 -> 1/2 scale (16 ch)
        self.enc1 = features[1]  # Stride 2 -> 1/4 scale (16 ch)
        self.enc2 = features[2:4]  # Stride 2 at idx 2 -> 1/8 scale (24 ch)
        self.enc3 = features[4:9]  # Stride 2 at idx 4 -> 1/16 scale (48 ch)
        self.enc4 = features[9:]  # Stride 2 at idx 9 -> 1/32 scale (576 ch at idx 12)

        # ----------------------------------------------------------------
        # 3. Define Decoder Stages
        # ----------------------------------------------------------------
        # Decoder 4: Up(1/32) + Skip(1/16) -> 1/16
        # In: 576 (enc4) + 48 (enc3)
        self.dec4 = DecoderBlock(in_c=576, skip_c=48, out_c=256)

        # Decoder 3: Up(1/16) + Skip(1/8) -> 1/8
        # In: 256 (dec4) + 24 (enc2)
        self.dec3 = DecoderBlock(in_c=256, skip_c=24, out_c=128)

        # Decoder 2: Up(1/8) + Skip(1/4) -> 1/4
        # In: 128 (dec3) + 16 (enc1)
        self.dec2 = DecoderBlock(in_c=128, skip_c=16, out_c=64)

        # Decoder 1: Up(1/4) + Skip(1/2) -> 1/2
        # In: 64 (dec2) + 16 (enc0)
        self.dec1 = DecoderBlock(in_c=64, skip_c=16, out_c=32)

        # ----------------------------------------------------------------
        # 4. Final Head
        # ----------------------------------------------------------------
        # Upsample 1/2 -> 1/1
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x0 = self.enc0(x)  # 1/2, 16ch
        x1 = self.enc1(x0)  # 1/4, 16ch
        x2 = self.enc2(x1)  # 1/8, 24ch
        x3 = self.enc3(x2)  # 1/16, 48ch
        x4 = self.enc4(x3)  # 1/32, 576ch

        # --- Decoder ---
        d4 = self.dec4(x4, x3)  # -> 1/16
        d3 = self.dec3(d4, x2)  # -> 1/8
        d2 = self.dec2(d3, x1)  # -> 1/4
        d1 = self.dec1(d2, x0)  # -> 1/2

        # --- Head ---
        out = self.final_up(d1)  # -> 1/1
        out = self.final_conv(out)

        return out
