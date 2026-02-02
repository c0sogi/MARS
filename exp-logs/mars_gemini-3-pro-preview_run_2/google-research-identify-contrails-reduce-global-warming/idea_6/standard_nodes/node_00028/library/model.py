import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class SCSEModule(nn.Module):
    """
    Spatial and Channel Squeeze & Excitation Block.
    Enhances important features by recalibrating channel-wise and spatial-wise feature maps.
    This helps suppress noise in the skip connections.
    """

    def __init__(self, in_channels, reduction=16):
        super(SCSEModule, self).__init__()
        # Channel Squeeze and Excitation
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, in_channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, 1),
            nn.Sigmoid(),
        )
        # Spatial Squeeze and Excitation
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # Combine channel and spatial attention
        return x * self.cSE(x) + x * self.sSE(x)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    Captures multi-scale context using dilated convolutions with different rates.
    This provides isotropic context aggregation, which is superior to axis-aligned
    pooling for features with arbitrary orientations (Cite Lesson 00025).
    """

    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=rates[0],
                dilation=rates[0],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=rates[1],
                dilation=rates[1],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=rates[2],
                dilation=rates[2],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv_pool(self.avg_pool(x))
        x5 = F.interpolate(x5, size=size, mode="bilinear", align_corners=False)

        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.project(out)


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block with Bilinear Upsampling and SCSE Attention.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Input channels = Upsampled Input + Skip Connection
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.scse = SCSEModule(out_channels)

    def forward(self, x, skip=None):
        # Bilinear Upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if skip is not None:
            # Handle slight dimension mismatches due to padding/pooling in encoder
            if x.size() != skip.size():
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.scse(x)
        return x


class StripPoolingResNet18UNet(nn.Module):
    """
    U-Net architecture with ResNet18 encoder and Strip Pooling bottleneck.
    Designed for contrail segmentation.
    """

    def __init__(self, in_channels=Config.IN_CHANNELS, pretrained=True):
        super(StripPoolingResNet18UNet, self).__init__()

        # ===========================
        # Encoder (ResNet18)
        # ===========================
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = models.resnet18(weights=weights)

        # Modify first layer to accept 6 channels (3 Ash + 3 Temporal)
        # ResNet18 conv1: 7x7, stride 2, padding 3
        if in_channels != 3:
            original_conv1 = self.encoder.conv1
            self.encoder.conv1 = nn.Conv2d(
                in_channels,
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias,
            )

            # Initialize new weights
            if pretrained:
                with torch.no_grad():
                    # Copy RGB weights to the first 3 channels
                    self.encoder.conv1.weight[:, :3] = original_conv1.weight
                    # For channels 3-6 (Temporal), we leave them with default init
                    # or could initialize them similarly. Default Kaiming is acceptable.

        # ===========================
        # Bottleneck (Strip Pooling)
        # ===========================
        # ResNet18 Layer 4 output channels: 512
        self.bottleneck_spm = StripPooling(512)
        # Post-SPM convolution to mix features
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ===========================
        # Decoder
        # ===========================
        # Layer 4 (512) -> Dec4 -> 256. Skip: Layer 3 (256)
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)

        # Layer 3 (256) -> Dec3 -> 128. Skip: Layer 2 (128)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)

        # Layer 2 (128) -> Dec2 -> 64. Skip: Layer 1 (64)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)

        # Layer 1 (64) -> Dec1 -> 64. Skip: Layer 0 (64, before MaxPool)
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        # Final Upsampling and Head
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        # ===========================
        # Encoder
        # ===========================
        # Input: B x 6 x 256 x 256
        x0 = self.encoder.conv1(x)  # -> 128x128, 64ch
        x0 = self.encoder.bn1(x0)
        x0 = self.encoder.relu(x0)  # Skip for Dec1

        x1 = self.encoder.maxpool(x0)  # -> 64x64, 64ch
        x1 = self.encoder.layer1(x1)  # -> 64x64, 64ch (Skip for Dec2)

        x2 = self.encoder.layer2(x1)  # -> 32x32, 128ch (Skip for Dec3)

        x3 = self.encoder.layer3(x2)  # -> 16x16, 256ch (Skip for Dec4)

        x4 = self.encoder.layer4(x3)  # -> 8x8, 512ch

        # ===========================
        # Bottleneck
        # ===========================
        b = self.bottleneck_spm(x4)
        b = self.bottleneck_conv(b)

        # ===========================
        # Decoder
        # ===========================
        d4 = self.dec4(b, x3)  # 8->16
        d3 = self.dec3(d4, x2)  # 16->32
        d2 = self.dec2(d3, x1)  # 32->64
        d1 = self.dec1(d2, x0)  # 64->128

        # ===========================
        # Head
        # ===========================
        out = self.final_conv(d1)  # 128->256

        return out
