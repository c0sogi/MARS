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


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module.
    Factorizes attention into two parallel 1D feature encodings to capture
    long-range dependencies along one spatial direction while preserving
    precise positional information along the other.
    """

    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Pool
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate along spatial dimension
        y = torch.cat([x_h, x_w], dim=2)

        # Shared 1x1 Conv
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Output 1x1 Convs
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


class DecoderBlock(nn.Module):
    """
    Decoder Block with Bilinear Upsampling and Coordinate Attention.
    Performs upsampling, concatenation with skip connection, convolution,
    and then applies attention.
    """

    def __init__(self, in_channels, skip_channels, out_channels, use_coord_att=True):
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

        self.use_coord_att = use_coord_att
        if use_coord_att:
            self.att = CoordinateAttention(out_channels)

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
        if self.use_coord_att:
            x = self.att(x)

        return x
