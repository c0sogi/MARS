import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class DepthStem(nn.Module):
    """
    Projects the input volume depth to 3 channels using 1x1 convolution
    and applies Instance Normalization.
    """

    def __init__(self, in_channels=65, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # Instance Norm for global intensity invariance per sample
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class AtrousBottleneck(nn.Module):
    """
    Bottleneck with parallel dilated convolutions (rates 1, 2, 4, 8)
    to capture multi-scale context.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 4

        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=4,
                dilation=4,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=8,
                dilation=8,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(mid_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.project(out)
        return out


class DecoderBlock(nn.Module):
    """
    Standard U-Net decoder block with skip connections.
    Uses Batch Normalization for spatial consistency.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
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
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        if skip is not None:
            # Handle potential padding discrepancies
            if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                x = F.interpolate(
                    x,
                    size=(skip.size(2), skip.size(3)),
                    mode="bilinear",
                    align_corners=True,
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class InkUNet(nn.Module):
    """
    U-Net architecture for Ink Detection.
    - Input: (B, 65, H, W)
    - Output: (B, 1, H, W) logits
    """

    def __init__(self, z_dim=65):
        super().__init__()

        # 1. Depth Compression Stem
        self.stem = DepthStem(in_channels=z_dim, out_channels=3)

        # 2. Encoder (ResNet18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Extract layers
        self.enc_layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu
        )  # Output: 64, H/2, W/2

        self.maxpool = resnet.maxpool  # Output: 64, H/4, W/4

        self.enc_layer1 = resnet.layer1  # Output: 64, H/4, W/4
        self.enc_layer2 = resnet.layer2  # Output: 128, H/8, W/8
        self.enc_layer3 = resnet.layer3  # Output: 256, H/16, W/16
        self.enc_layer4 = resnet.layer4  # Output: 512, H/32, W/32

        # 3. Atrous Bottleneck
        self.bottleneck = AtrousBottleneck(in_channels=512, out_channels=512)

        # 4. Decoder
        # Dec4: Input 512 (bottleneck) + Skip 256 (layer3) -> Out 256
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)

        # Dec3: Input 256 + Skip 128 (layer2) -> Out 128
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)

        # Dec2: Input 128 + Skip 64 (layer1) -> Out 64
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)

        # Dec1: Input 64 + Skip 64 (layer0 pre-pool) -> Out 32
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)

        # Final Upsample to original resolution
        self.final_conv = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # H/2 -> H
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: (B, 65, H, W)

        # Stem
        x = self.stem(x)  # (B, 3, H, W)

        # Encoder
        x0 = self.enc_layer0(x)  # (B, 64, H/2, W/2)
        x_mp = self.maxpool(x0)  # (B, 64, H/4, W/4)

        x1 = self.enc_layer1(x_mp)  # (B, 64, H/4, W/4)
        x2 = self.enc_layer2(x1)  # (B, 128, H/8, W/8)
        x3 = self.enc_layer3(x2)  # (B, 256, H/16, W/16)
        x4 = self.enc_layer4(x3)  # (B, 512, H/32, W/32)

        # Bottleneck
        bn = self.bottleneck(x4)  # (B, 512, H/32, W/32)

        # Decoder
        d4 = self.dec4(bn, x3)  # (B, 256, H/16, W/16)
        d3 = self.dec3(d4, x2)  # (B, 128, H/8, W/8)
        d2 = self.dec2(d3, x1)  # (B, 64, H/4, W/4)
        d1 = self.dec1(d2, x0)  # (B, 32, H/2, W/2)

        # Final
        out = self.final_conv(d1)  # (B, 1, H, W)

        return out
