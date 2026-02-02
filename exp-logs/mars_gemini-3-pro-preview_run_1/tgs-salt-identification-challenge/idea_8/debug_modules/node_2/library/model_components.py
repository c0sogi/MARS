import torch
import torch.nn as nn
import torch.nn.functional as F


class SCSEBlock(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE) Block.
    Recalibrates feature maps spatially and channel-wise to highlight relevant features.
    Reference: Roy et al., "Concurrent Spatial and Channel Squeeze & Excitation..."
    """

    def __init__(self, in_channels, reduction=16):
        super(SCSEBlock, self).__init__()

        # Channel Squeeze & Excitation (cSE)
        # Global Average Pooling -> Dense -> ReLU -> Dense -> Sigmoid
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Ensure reduction doesn't make channels < 1
        reduced_channels = max(1, in_channels // reduction)

        self.channel_excitation = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Spatial Squeeze & Excitation (sSE)
        # 1x1 Conv -> Sigmoid
        self.spatial_excitation = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        # cSE path
        chn_se = self.avg_pool(x)
        chn_se = self.channel_excitation(chn_se)
        out_chn = x * chn_se

        # sSE path
        spa_se = self.spatial_excitation(x)
        out_spa = x * spa_se

        # Concurrent combination (Addition)
        return out_chn + out_spa


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with two 3x3 convolutions and a skip connection.
    Used for the encoder backbone.
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
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.downsample = None
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


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    Captures multi-scale context using parallel dilated convolutions.
    """

    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()

        modules = []

        # 1x1 Convolution
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Dilated Convolutions (Rates: 6, 12, 18)
        for rate in rates[1:]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Image Pooling (Global Context)
        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.convs = nn.ModuleList(modules)

        # Project concatenated features
        # Input channels: (number of rates + 1 for image pooling) * out_channels
        self.project = nn.Sequential(
            nn.Conv2d(
                len(rates) * out_channels + out_channels, out_channels, 1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []

        # Apply dilated convs
        for conv in self.convs:
            res.append(conv(x))

        # Apply image pooling and upsample back to input size
        pool = self.image_pooling(x)
        pool = F.interpolate(
            pool, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        res.append(pool)

        # Concatenate all branches
        res = torch.cat(res, dim=1)

        return self.project(res)


class DecoderBlock(nn.Module):
    """
    Decoder Block for U-Net.
    Performs bilinear upsampling, concatenation with skip connection,
    convolutional processing, and attention via scSE.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Input channels = Up-sampled channels + Skip connection channels
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

        # Attention Mechanism
        self.scse = SCSEBlock(out_channels)

    def forward(self, x, skip=None):
        # 1. Bilinear Upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # 2. Concatenate with Skip Connection
        if skip is not None:
            # Handle potential shape mismatch due to padding/rounding
            if x.size() != skip.size():
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        # 3. Convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # 4. Attention
        x = self.scse(x)

        return x
