import torch
import torch.nn as nn
import torch.nn.functional as F


class SCSEBlock(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE) Block.
    Recalibrates feature maps spatially and channel-wise.
    """

    def __init__(self, in_channels, reduction=16):
        super(SCSEBlock, self).__init__()
        # Channel Squeeze and Excitation (cSE)
        # Global Average Pooling -> Dense -> ReLU -> Dense -> Sigmoid
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

        # Spatial Squeeze and Excitation (sSE)
        # Conv1x1 -> Sigmoid
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # Concurrent: x * cSE + x * sSE
        return x * self.cSE(x) + x * self.sSE(x)


class ResidualBlock(nn.Module):
    """
    Standard Residual Block: Conv3x3-BN-ReLU-Conv3x3-BN + Skip Connection.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        # If stride > 1 or channels change, apply 1x1 conv to skip connection
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class PPM(nn.Module):
    """
    Pyramid Pooling Module (PPM).
    Captures global context by pooling at multiple scales.
    """

    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes

        # Reduction dimension for each pool branch
        reduction_dim = in_channels // 4

        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=s),
                    nn.Conv2d(in_channels, reduction_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(reduction_dim),
                    nn.ReLU(inplace=True),
                )
                for s in pool_sizes
            ]
        )

        # Final bottleneck to fuse original features + pooled features
        # Input channels = Original + 4 * (Original // 4) = 2 * Original
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + len(pool_sizes) * reduction_dim,
                in_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        ppm_outs = [x]

        for stage in self.stages:
            out = stage(x)
            # Bilinear upsample to match input size
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
            ppm_outs.append(out)

        out = torch.cat(ppm_outs, dim=1)
        out = self.bottleneck(out)
        return out


class DecoderBlock(nn.Module):
    """
    Decoder Block: Upsample -> Concat -> ResidualBlock -> SCSE.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # We use a ResidualBlock for the convolution part
        self.conv = ResidualBlock(in_channels + skip_channels, out_channels)
        self.scse = SCSEBlock(out_channels)

    def forward(self, x, skip):
        # Bilinear Upsampling
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        # Concatenate with skip connection
        out = torch.cat([x, skip], dim=1)
        # Convolve
        out = self.conv(out)
        # Attention
        out = self.scse(out)
        return out


class ResUNetPPM(nn.Module):
    """
    Context-Aware Deep Residual U-Net with PPM and Deep Supervision.
    """

    def __init__(self, in_channels=2, num_classes=1, filters=64):
        super(ResUNetPPM, self).__init__()

        # --- Encoder ---
        # Initial Conv: 128x128
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
        )

        # Layer 1: 128x128 -> 64x64
        self.layer1 = ResidualBlock(filters, filters * 2, stride=2)
        # Layer 2: 64x64 -> 32x32
        self.layer2 = ResidualBlock(filters * 2, filters * 4, stride=2)
        # Layer 3: 32x32 -> 16x16
        self.layer3 = ResidualBlock(filters * 4, filters * 8, stride=2)
        # Layer 4 (Bridge part 1): 16x16 -> 8x8
        self.layer4 = ResidualBlock(filters * 8, filters * 8, stride=2)

        # --- Bottleneck ---
        # PPM at Bridge: 8x8
        self.ppm = PPM(filters * 8)  # 512 channels

        # --- Decoder ---
        # Up 1: 8x8 -> 16x16. Skip: layer3 (512). Input: 512 (from PPM).
        self.up1 = DecoderBlock(filters * 8, filters * 8, filters * 4)  # Out: 256

        # Up 2: 16x16 -> 32x32. Skip: layer2 (256). Input: 256.
        self.up2 = DecoderBlock(filters * 4, filters * 4, filters * 2)  # Out: 128

        # Up 3: 32x32 -> 64x64. Skip: layer1 (128). Input: 128.
        self.up3 = DecoderBlock(filters * 2, filters * 2, filters)  # Out: 64

        # Up 4: 64x64 -> 128x128. Skip: init_conv (64). Input: 64.
        self.up4 = DecoderBlock(filters, filters, filters)  # Out: 64

        # --- Heads ---
        # Final segmentation head
        self.final_conv = nn.Conv2d(filters, num_classes, kernel_size=1)

        # Deep Supervision Heads (Auxiliary)
        # Aux 1 at 1/8 scale (16x16)
        self.aux1 = nn.Conv2d(filters * 4, num_classes, kernel_size=1)
        # Aux 2 at 1/4 scale (32x32)
        self.aux2 = nn.Conv2d(filters * 2, num_classes, kernel_size=1)
        # Aux 3 at 1/2 scale (64x64)
        self.aux3 = nn.Conv2d(filters, num_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder Path ---
        x0 = self.init_conv(x)  # 64, 128, 128
        x1 = self.layer1(x0)  # 128, 64, 64
        x2 = self.layer2(x1)  # 256, 32, 32
        x3 = self.layer3(x2)  # 512, 16, 16
        x4 = self.layer4(x3)  # 512, 8, 8

        # --- Bottleneck ---
        x_ppm = self.ppm(x4)  # 512, 8, 8

        # --- Decoder Path ---
        d1 = self.up1(x_ppm, x3)  # 256, 16, 16
        d2 = self.up2(d1, x2)  # 128, 32, 32
        d3 = self.up3(d2, x1)  # 64, 64, 64
        d4 = self.up4(d3, x0)  # 64, 128, 128

        # --- Heads ---
        logits = self.final_conv(d4)

        # Deep Supervision: Return aux outputs during training
        if self.training:
            # Upsample aux outputs to match logits size (128x128) for loss calculation
            # Note: Some implementations calculate loss at lower res, but upsampling
            # simplifies the loss function interface.

            aux1 = self.aux1(d1)
            aux1 = F.interpolate(
                aux1, size=logits.shape[2:], mode="bilinear", align_corners=True
            )

            aux2 = self.aux2(d2)
            aux2 = F.interpolate(
                aux2, size=logits.shape[2:], mode="bilinear", align_corners=True
            )

            aux3 = self.aux3(d3)
            aux3 = F.interpolate(
                aux3, size=logits.shape[2:], mode="bilinear", align_corners=True
            )

            return logits, aux1, aux2, aux3

        return logits
