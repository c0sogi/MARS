import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concat -> ConvBlock
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Input to conv is (upsampled_channels + skip_channels)
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

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if skip is not None:
            # Handle potential shape mismatches (e.g., odd dimensions) via interpolation
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class MIPUNet(nn.Module):
    """
    MIP U-Net: ResNet18 Encoder + Custom Decoder for 2D MIP Segmentation.
    Expects a single-channel 2D Maximum Intensity Projection as input.
    """

    def __init__(
        self,
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    ):
        super().__init__()

        # --- Encoder (ResNet18/34) ---
        if encoder_name == "resnet18":
            weights = (
                models.ResNet18_Weights.DEFAULT
                if encoder_weights == "imagenet"
                else None
            )
            self.encoder = models.resnet18(weights=weights)
        elif encoder_name == "resnet34":
            weights = (
                models.ResNet34_Weights.DEFAULT
                if encoder_weights == "imagenet"
                else None
            )
            self.encoder = models.resnet34(weights=weights)
        else:
            raise ValueError(
                "MIPUNet currently only supports 'resnet18' and 'resnet34' encoders."
            )

        # Modify first layer for 1-channel input (Grayscale MIP) if necessary
        if in_channels != 3:
            original_conv = self.encoder.conv1
            self.encoder.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

            # Initialize with average of pretrained weights to preserve feature detection
            if weights is not None:
                with torch.no_grad():
                    self.encoder.conv1.weight[:] = torch.mean(
                        original_conv.weight, dim=1, keepdim=True
                    )

        # Encoder Channel Sizes for ResNet18:
        # layer0 (conv1+bn+relu): 64
        # layer1: 64
        # layer2: 128
        # layer3: 256
        # layer4: 512 (Bottleneck)

        # --- Decoder ---
        # Block 4: Process layer4 (512) + skip layer3 (256) -> 256
        self.decoder4 = DecoderBlock(
            in_channels=512, skip_channels=256, out_channels=256
        )

        # Block 3: Process decoder4 (256) + skip layer2 (128) -> 128
        self.decoder3 = DecoderBlock(
            in_channels=256, skip_channels=128, out_channels=128
        )

        # Block 2: Process decoder3 (128) + skip layer1 (64) -> 64
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)

        # Block 1: Process decoder2 (64) + skip layer0 (64) -> 64
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        # --- Final Head ---
        # Upsample from H/2 to H and project to classes
        self.final_upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, classes, kernel_size=1),
        )

    def forward(self, x):
        # --- Encoder Forward ---
        # Stem
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x0 = self.encoder.relu(x)  # (B, 64, H/2, W/2) -> Skip for Decoder 1

        x = self.encoder.maxpool(x0)
        x1 = self.encoder.layer1(x)  # (B, 64, H/4, W/4) -> Skip for Decoder 2
        x2 = self.encoder.layer2(x1)  # (B, 128, H/8, W/8) -> Skip for Decoder 3
        x3 = self.encoder.layer3(x2)  # (B, 256, H/16, W/16) -> Skip for Decoder 4
        x4 = self.encoder.layer4(x3)  # (B, 512, H/32, W/32) -> Bottleneck

        # --- Decoder Forward ---
        d4 = self.decoder4(x4, x3)  # -> (B, 256, H/16, W/16)
        d3 = self.decoder3(d4, x2)  # -> (B, 128, H/8, W/8)
        d2 = self.decoder2(d3, x1)  # -> (B, 64, H/4, W/4)
        d1 = self.decoder1(d2, x0)  # -> (B, 64, H/2, W/2)

        # --- Final Head ---
        out = self.final_upsample(d1)  # -> (B, 64, H, W)
        logits = self.final_conv(out)  # -> (B, classes, H, W)

        return logits
