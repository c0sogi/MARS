import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Standard ResNet Basic Block for the encoder.
    Consists of two 3x3 convolutions with Batch Normalization and ReLU.
    Includes a skip connection handling stride and channel changes.
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


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE).
    Cite solution_lesson_node_00055: Outperforms Coordinate Attention on small images.
    Cite solution_lesson_node_00007: Enhances feature recalibration.
    """

    def __init__(self, in_channels, reduction=16):
        super(SCSEModule, self).__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, in_channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class DecoderBlock(nn.Module):
    """
    Decoder Block with Bilinear Upsampling and scSE Attention.
    Performs upsampling, concatenation with skip connection, convolution,
    and then applies attention.
    """

    def __init__(self, in_channels, skip_channels, out_channels, use_scse=True):
        super(DecoderBlock, self).__init__()

        # We use bilinear upsampling in forward, so no layer definition needed here
        # unless using nn.Upsample. Functional interpolate is preferred for flexibility.

        # Convolution block to process fused features
        # Input channels = upsampled channels (in_channels) + skip channels
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

        self.use_scse = use_scse
        if use_scse:
            self.att = SCSEModule(out_channels)

    def forward(self, x, skip=None):
        # Bilinear Upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Handle Skip Connection
        if skip is not None:
            # If dimensions don't match exactly due to odd sizes, resize x to match skip
            if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                x = F.interpolate(
                    x,
                    size=(skip.size(2), skip.size(3)),
                    mode="bilinear",
                    align_corners=True,
                )
            x = torch.cat([x, skip], dim=1)

        # Convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Attention
        if self.use_scse:
            x = self.att(x)

        return x
