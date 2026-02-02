import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class VolumetricStem(nn.Module):
    """
    Shallow 3D Convolutional Stem to process the Z-dimension of the input volume.
    Reduces depth while extracting features, then flattens to 2D.
    """

    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()

        # Layer 1: Preserve Z resolution, extract low-level 3D features
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Layer 2: Downsample Z by 2
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                base_channels,
                base_channels * 2,
                kernel_size=3,
                stride=(2, 1, 1),
                padding=1,
            ),
            nn.InstanceNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        # Layer 3: Downsample Z by 2
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=3,
                stride=(2, 1, 1),
                padding=1,
            ),
            nn.InstanceNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        # Calculate output channels dynamically based on Z_DIM
        self.out_channels = self._get_flattened_channels(in_channels)

    def _get_flattened_channels(self, in_channels):
        # Pass a dummy tensor to calculate the exact output size after 3D convs
        dummy = torch.zeros(1, in_channels, Config.Z_DIM, 32, 32)
        with torch.no_grad():
            x = self.conv1(dummy)
            x = self.conv2(x)
            x = self.conv3(x)
        # Output shape is (B, C, D, H, W). We flatten C*D.
        return x.shape[1] * x.shape[2]

    def forward(self, x):
        # x: (B, 1, Z, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten Z into Channel dimension
        # (B, C, D, H, W) -> (B, C*D, H, W)
        b, c, d, h, w = x.shape
        x = x.view(b, c * d, h, w)
        return x


class DilatedBottleneck(nn.Module):
    """
    Bottleneck with exponentially increasing dilation rates to expand receptive field.
    """

    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, mid_channels, 3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm2d(mid_channels)

        self.conv4 = nn.Conv2d(mid_channels, mid_channels, 3, padding=8, dilation=8)
        self.bn4 = nn.BatchNorm2d(mid_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Sequential application
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        return x


class DecoderBlock(nn.Module):
    """
    Standard U-Net decoder block: Upsample -> Concat -> Conv -> Conv.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        # Bilinear upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        if skip is not None:
            # Handle potential shape mismatch due to padding/cropping in encoder
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class Hybrid3D2DUNet(nn.Module):
    """
    Hybrid architecture: 3D Stem -> 2D ResNet34 Encoder -> Dilated Bottleneck -> U-Net Decoder.
    """

    def __init__(self):
        super().__init__()

        # 1. Volumetric Stem
        self.stem = VolumetricStem(in_channels=Config.IN_CHANNELS)

        # 2. Encoder (ResNet34)
        # Load pretrained weights
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Adapter: Transform flattened 3D features to ResNet input (64 channels)
        # ResNet conv1 is usually 7x7 stride 2. We mimic this geometry but change input channels.
        self.adapter = nn.Conv2d(
            self.stem.out_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Encoder Layers
        # enc1: (B, 64, H/2, W/2)
        self.enc1 = nn.Sequential(self.adapter, resnet.bn1, resnet.relu)
        # pool: (B, 64, H/4, W/4)
        self.pool = resnet.maxpool
        # enc2: (B, 64, H/4, W/4)
        self.enc2 = resnet.layer1
        # enc3: (B, 128, H/8, W/8)
        self.enc3 = resnet.layer2
        # enc4: (B, 256, H/16, W/16)
        self.enc4 = resnet.layer3
        # enc5: (B, 512, H/32, W/32)
        self.enc5 = resnet.layer4

        # 3. Bottleneck
        self.bottleneck = DilatedBottleneck(512, 512)

        # 4. Decoder
        # d4: Input 512 + Skip 256 -> Out 256
        self.dec4 = DecoderBlock(512, 256, 256)
        # d3: Input 256 + Skip 128 -> Out 128
        self.dec3 = DecoderBlock(256, 128, 128)
        # d2: Input 128 + Skip 64 -> Out 64
        self.dec2 = DecoderBlock(128, 64, 64)
        # d1: Input 64 + Skip 64 -> Out 64
        self.dec1 = DecoderBlock(64, 64, 64)

        # 5. Final Output Head
        # Upsample from H/2 to H
        self.final_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, Config.OUT_CHANNELS, 1),
        )

    def forward(self, x):
        # x: (B, 1, 65, H, W)

        # --- Stem ---
        x = self.stem(x)  # (B, C_flat, H, W)

        # --- Encoder ---
        e1 = self.enc1(x)  # (B, 64, H/2, W/2)
        p1 = self.pool(e1)  # (B, 64, H/4, W/4)
        e2 = self.enc2(p1)  # (B, 64, H/4, W/4)
        e3 = self.enc3(e2)  # (B, 128, H/8, W/8)
        e4 = self.enc4(e3)  # (B, 256, H/16, W/16)
        e5 = self.enc5(e4)  # (B, 512, H/32, W/32)

        # --- Bottleneck ---
        b = self.bottleneck(e5)  # (B, 512, H/32, W/32)

        # --- Decoder ---
        d4 = self.dec4(b, e4)  # (B, 256, H/16, W/16)
        d3 = self.dec3(d4, e3)  # (B, 128, H/8, W/8)
        d2 = self.dec2(d3, e2)  # (B, 64, H/4, W/4)
        d1 = self.dec1(d2, e1)  # (B, 64, H/2, W/2)

        # --- Output ---
        out = self.final_head(d1)  # (B, 1, H, W)

        return out
