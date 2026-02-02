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


class ResNetUNet(nn.Module):
    """
    U-Net with ResNet18 Encoder for Contrail Segmentation.
    Cite solution_lesson_node_00002: Scale and Capacity are Prerequisites.

    Input: 6 Channels (Ash t=4, Difference t=4-t=3)
    Output: 1 Channel (Logits)
    """

    def __init__(self, in_channels=Config.IN_CHANNELS, num_classes=1):
        super().__init__()

        # Load Pretrained Backbone
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")

        # Modify first layer for 6 channels
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            self.backbone.conv1.weight[:, :3] = original_conv.weight
            if in_channels > 3:
                self.backbone.conv1.weight[:, 3:6] = original_conv.weight

        # Encoder layers
        self.enc0 = nn.Sequential(
            self.backbone.conv1, self.backbone.bn1, self.backbone.relu
        )  # 1/2, 64
        self.pool = self.backbone.maxpool  # 1/4, 64
        self.enc1 = self.backbone.layer1  # 1/4, 64
        self.enc2 = self.backbone.layer2  # 1/8, 128
        self.enc3 = self.backbone.layer3  # 1/16, 256
        self.enc4 = self.backbone.layer4  # 1/32, 512

        # Decoder
        # dec4: in=512, skip=256 -> out=256
        self.dec4 = DecoderBlock(in_c=512, skip_c=256, out_c=256)

        # dec3: in=256, skip=128 -> out=128
        self.dec3 = DecoderBlock(in_c=256, skip_c=128, out_c=128)

        # dec2: in=128, skip=64 -> out=64
        self.dec2 = DecoderBlock(in_c=128, skip_c=64, out_c=64)

        # dec1: in=64, skip=64 -> out=32
        self.dec1 = DecoderBlock(in_c=64, skip_c=64, out_c=32)

        # Final Head
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x0 = self.enc0(x)  # 1/2, 64
        x_pool = self.pool(x0)  # 1/4, 64
        x1 = self.enc1(x_pool)  # 1/4, 64
        x2 = self.enc2(x1)  # 1/8, 128
        x3 = self.enc3(x2)  # 1/16, 256
        x4 = self.enc4(x3)  # 1/32, 512

        # --- Decoder ---
        d4 = self.dec4(x4, x3)  # -> 1/16
        d3 = self.dec3(d4, x2)  # -> 1/8
        d2 = self.dec2(d3, x1)  # -> 1/4
        d1 = self.dec1(d2, x0)  # -> 1/2

        # --- Head ---
        out = self.final_up(d1)  # -> 1/1
        out = self.final_conv(out)

        return out
