import torch
import torch.nn as nn
import torch.nn.functional as F


class SCSEBlock(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE) Block.
    Enhances important features by reweighting spatially and channel-wise.
    """

    def __init__(self, channels, reduction=16):
        super(SCSEBlock, self).__init__()
        # Channel Squeeze and Excitation (cSE)
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

        # Spatial Squeeze and Excitation (sSE)
        self.sSE = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # Concurrent combination: Input * cSE + Input * sSE
        return x * self.cSE(x) + x * self.sSE(x)


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with two 3x3 convolutions and a skip connection.
    Handles channel changing or downsampling via a 1x1 projection in the skip path.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class DeepSupervisionHead(nn.Module):
    """
    Auxiliary prediction head for Deep Supervision.
    Maps feature maps to binary segmentation logits.
    """

    def __init__(self, in_channels, out_channels):
        super(DeepSupervisionHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DeepResUNet(nn.Module):
    """
    Deep Residual U-Net with SCSE blocks and Deep Supervision.

    Args:
        in_channels (int): Number of image channels (default 1 for grayscale).
        out_channels (int): Number of output classes (default 1 for binary mask).
        depth_fused (bool): Whether to expect an additional depth channel fused at input.
    """

    def __init__(self, in_channels=1, out_channels=1, depth_fused=True):
        super(DeepResUNet, self).__init__()

        # If depth is fused, the input tensor will have one extra channel
        input_dim = in_channels + 1 if depth_fused else in_channels

        # --- Encoder ---
        # Stem: Project input to 64 channels
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Encoder Stages
        # Input: 128x128 -> Output: 128x128
        self.enc1 = ResidualBlock(64, 64)

        # Pooling 1: 128 -> 64
        self.pool1 = nn.MaxPool2d(2, 2)
        # Stage 2
        self.enc2 = ResidualBlock(64, 128)

        # Pooling 2: 64 -> 32
        self.pool2 = nn.MaxPool2d(2, 2)
        # Stage 3
        self.enc3 = ResidualBlock(128, 256)

        # Pooling 3: 32 -> 16
        self.pool3 = nn.MaxPool2d(2, 2)
        # Stage 4
        self.enc4 = ResidualBlock(256, 512)

        # Pooling 4: 16 -> 8
        self.pool4 = nn.MaxPool2d(2, 2)

        # --- Bridge ---
        # Center block at lowest resolution (8x8)
        self.center = ResidualBlock(512, 1024)

        # --- Decoder ---
        # Upsample 4: 8 -> 16
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(512 + 512, 512)  # Concat with enc4
        self.scse4 = SCSEBlock(512)

        # Upsample 3: 16 -> 32
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(256 + 256, 256)  # Concat with enc3
        self.scse3 = SCSEBlock(256)
        # Deep Supervision Head at 32x32
        self.aux_head_32 = DeepSupervisionHead(256, out_channels)

        # Upsample 2: 32 -> 64
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(128 + 128, 128)  # Concat with enc2
        self.scse2 = SCSEBlock(128)
        # Deep Supervision Head at 64x64
        self.aux_head_64 = DeepSupervisionHead(128, out_channels)

        # Upsample 1: 64 -> 128
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(64 + 64, 64)  # Concat with enc1
        self.scse1 = SCSEBlock(64)
        # Final Head at 128x128
        self.final_head = DeepSupervisionHead(64, out_channels)

    def forward(self, x):
        # --- Encoder Forward ---
        x_stem = self.stem(x)

        e1 = self.enc1(x_stem)  # 128x128, 64 ch
        p1 = self.pool1(e1)  # 64x64

        e2 = self.enc2(p1)  # 64x64, 128 ch
        p2 = self.pool2(e2)  # 32x32

        e3 = self.enc3(p2)  # 32x32, 256 ch
        p3 = self.pool3(e3)  # 16x16

        e4 = self.enc4(p3)  # 16x16, 512 ch
        p4 = self.pool4(e4)  # 8x8

        # --- Bridge Forward ---
        center = self.center(p4)  # 8x8, 1024 ch

        # --- Decoder Forward ---
        # Block 4
        d4 = self.up4(center)  # 16x16, 512 ch
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d4 = self.scse4(d4)

        # Block 3
        d3 = self.up3(d4)  # 32x32, 256 ch
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d3 = self.scse3(d3)
        # Aux Output 32
        out32 = self.aux_head_32(d3)

        # Block 2
        d2 = self.up2(d3)  # 64x64, 128 ch
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d2 = self.scse2(d2)
        # Aux Output 64
        out64 = self.aux_head_64(d2)

        # Block 1
        d1 = self.up1(d2)  # 128x128, 64 ch
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        d1 = self.scse1(d1)
        # Final Output 128
        out128 = self.final_head(d1)

        if self.training:
            # Return list for Deep Supervision Loss
            return [out128, out64, out32]
        else:
            # Return only final prediction during inference
            return out128
