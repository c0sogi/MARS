import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling Layer.

    Applies Max Pooling (Peaks) and Min Pooling (Shadows) in parallel
    and concatenates the outputs along the channel dimension.

    This preserves both the brightest features (strong scatterers) and
    the darkest features (shadows), expanding the channel depth by 2x.
    """

    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(DualPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        # Min pooling is implemented via max pooling on negated input
        # We use the same kernel/stride for alignment

    def forward(self, x):
        # Path 1: Max Pooling (Peaks)
        out_max = self.max_pool(x)

        # Path 2: Min Pooling (Shadows)
        # min(x) = -max(-x)
        out_min = -self.max_pool(-x)

        # Concatenate along channel dimension (dim=1)
        return torch.cat([out_max, out_min], dim=1)


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM.
    Uses Mixed Pooling (Avg + Max) to aggregate spatial information.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        # To save parameters and compute, we reduce channels by 'ratio'
        # Using Conv2d with kernel_size=1 is equivalent to Linear for this shape
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Average Pooling path
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # Global Max Pooling path
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        # Sum and activate
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for CBAM.
    Uses Mixed Pooling (Avg + Max) across channels to aggregate channel information.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        # Input channels = 2 (1 for avg pool, 1 for max pool)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Max pooling across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate along channel dimension
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # Convolution and activation
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies Channel Attention and Spatial Attention.
    """

    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply Channel Attention
        out = x * self.ca(x)
        # Apply Spatial Attention
        out = out * self.sa(out)
        return out


class WideConvBlock(nn.Module):
    """
    Wide Convolution Block for Delayed-Integration Backbone.

    Topology: 3x3 Conv -> BatchNorm -> ReLU.

    Designed to maintain high channel capacity or integrate features
    at the start of a receptive field block.
    """

    def __init__(self, in_channels, out_channels):
        super(WideConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
