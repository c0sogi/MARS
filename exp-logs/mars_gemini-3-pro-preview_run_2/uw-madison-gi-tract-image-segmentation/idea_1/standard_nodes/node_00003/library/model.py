import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block:
    Upsample -> Concatenate Skip -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Input channels to conv is sum of upsampled channels and skip channels
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        # x: Input from previous lower-resolution decoder layer
        # skip: Skip connection from corresponding encoder layer
        x = self.upsample(x)

        # Handle potential padding issues if dimensions are not perfect powers of 2
        # (Though with 256x256 input and standard strides, this shouldn't occur)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MobileNetV2UNet(nn.Module):
    """
    2.5D MobileNetV2 U-Net for Stomach and Intestine Segmentation.

    Encoder: MobileNetV2 (Pre-trained)
    Decoder: Standard U-Net Decoder with Skip Connections
    Head: 1x1 Conv + Sigmoid
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Load Backbone
        # Using weights="DEFAULT" maps to the best available weights (IMAGENET1K_V1 or V2)
        weights = "DEFAULT" if pretrained else None
        base_model = models.mobilenet_v2(weights=weights)
        self.encoder = base_model.features

        # Handle Input Channels (Adapt first layer if not 3 channels)
        if Config.IN_CHANNELS != 3:
            original_conv = self.encoder[0][0]
            self.encoder[0][0] = nn.Conv2d(
                Config.IN_CHANNELS,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False,
            )

        # Define Encoder Stages (Slicing MobileNetV2 features)
        # MobileNetV2 feature map indices and strides:
        # 0-1:   Stride 2, Channels 16   (128x128) -> enc1
        # 2-3:   Stride 4, Channels 24   (64x64)   -> enc2
        # 4-6:   Stride 8, Channels 32   (32x32)   -> enc3
        # 7-13:  Stride 16, Channels 96  (16x16)   -> enc4
        # 14-18: Stride 32, Channels 1280 (8x8)    -> enc5 (Bottleneck)

        self.enc1 = self.encoder[0:2]
        self.enc2 = self.encoder[2:4]
        self.enc3 = self.encoder[4:7]
        self.enc4 = self.encoder[7:14]
        self.enc5 = self.encoder[14:]

        # Define Decoder Stages
        # Arguments: in_channels, skip_channels, out_channels
        self.dec4 = DecoderBlock(1280, 96, 256)
        self.dec3 = DecoderBlock(256, 32, 128)
        self.dec2 = DecoderBlock(128, 24, 64)
        self.dec1 = DecoderBlock(64, 16, 32)

        # Final Upsampling Block (Stride 2 -> Stride 1)
        # Recovers 256x256 resolution from 128x128
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Segmentation Head
        self.head = nn.Conv2d(16, Config.NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x (torch.Tensor): Input image of shape (B, C, H, W)
        Returns:
            torch.Tensor: Predicted probabilities of shape (B, NumClasses, H, W)
        """
        # Encoder
        s1 = self.enc1(x)  # (B, 16, 128, 128)
        s2 = self.enc2(s1)  # (B, 24, 64, 64)
        s3 = self.enc3(s2)  # (B, 32, 32, 32)
        s4 = self.enc4(s3)  # (B, 96, 16, 16)
        s5 = self.enc5(s4)  # (B, 1280, 8, 8) - Bottleneck

        # Decoder with Skip Connections
        d4 = self.dec4(s5, s4)  # (B, 256, 16, 16)
        d3 = self.dec3(d4, s3)  # (B, 128, 32, 32)
        d2 = self.dec2(d3, s2)  # (B, 64, 64, 64)
        d1 = self.dec1(d2, s1)  # (B, 32, 128, 128)

        # Final Upsample to Original Resolution
        out = self.final_up(d1)  # (B, 16, 256, 256)

        # Prediction Head
        logits = self.head(out)  # (B, 3, 256, 256)

        # Return logits directly for numerical stability with BCEWithLogitsLoss
        return logits
