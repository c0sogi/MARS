import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from library.config import Config


class ConvBlock(nn.Module):
    """
    Standard building block for U-Net++ decoder nodes.
    Consists of two 3x3 convolutions, each followed by BatchNorm and ReLU.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class Unet(nn.Module):
    """
    2D U-Net with ResNet-34 Backbone.
    Standard 3-Channel Input.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone: ResNet34
        weights = ResNet34_Weights.IMAGENET1K_V1
        self.backbone = resnet34(weights=weights)

        # Filters: 64, 64, 128, 256, 512
        filters = [64, 64, 128, 256, 512]

        # 2. Decoder Blocks
        # Center: 512 -> 256
        self.center = ConvBlock(filters[4], filters[3])

        # Dec4: 256 + 256 -> 256
        self.dec4 = ConvBlock(filters[3] + filters[3], filters[3])

        # Dec3: 256 + 128 -> 128
        self.dec3 = ConvBlock(filters[3] + filters[2], filters[2])

        # Dec2: 128 + 64 -> 64
        self.dec2 = ConvBlock(filters[2] + filters[1], filters[1])

        # Dec1: 64 + 64 -> 64
        self.dec1 = ConvBlock(filters[1] + filters[0], filters[0])

        # Final Head
        self.final_head = nn.Conv2d(filters[0], Config.NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]

        # --- Encoder ---
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x0 = self.backbone.relu(x)  # Stride 2, 64

        x = self.backbone.maxpool(x0)
        x1 = self.backbone.layer1(x)  # Stride 4, 64
        x2 = self.backbone.layer2(x1)  # Stride 8, 128
        x3 = self.backbone.layer3(x2)  # Stride 16, 256
        x4 = self.backbone.layer4(x3)  # Stride 32, 512

        # --- Decoder ---
        # Center
        c = self.center(x4)  # 256

        # Dec4 (Up c + x3)
        d4 = self.dec4(
            torch.cat(
                [
                    F.interpolate(
                        c, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                    x3,
                ],
                dim=1,
            )
        )

        # Dec3 (Up d4 + x2)
        d3 = self.dec3(
            torch.cat(
                [
                    F.interpolate(
                        d4, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                    x2,
                ],
                dim=1,
            )
        )

        # Dec2 (Up d3 + x1)
        d2 = self.dec2(
            torch.cat(
                [
                    F.interpolate(
                        d3, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                    x1,
                ],
                dim=1,
            )
        )

        # Dec1 (Up d2 + x0)
        d1 = self.dec1(
            torch.cat(
                [
                    F.interpolate(
                        d2, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                    x0,
                ],
                dim=1,
            )
        )

        # Final Output
        logits = self.final_head(d1)

        # Upsample to original input size (Stride 2 -> 1)
        logits = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=True
        )

        return logits
