import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling Layer.

    Applies Max Pooling to capture peaks and Min Pooling to capture shadows (valleys).
    The outputs are concatenated along the channel dimension, expanding the depth (C -> 2C).

    Args:
        kernel_size (int): Size of the pooling window. Default: 2.
        stride (int): Stride of the pooling window. Default: 2.
    """

    def __init__(self, kernel_size=2, stride=2):
        super(DualPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # Max pooling captures high intensity signals (peaks)
        x_max = self.max_pool(x)

        # Min pooling captures low intensity signals (shadows)
        # Implemented as negative max pooling of negative input: min(x) = -max(-x)
        x_min = -self.max_pool(-x)

        # Concatenate along the channel dimension
        return torch.cat([x_max, x_min], dim=1)


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM.

    Uses Mixed Pooling (Average + Max) to aggregate spatial information, followed by
    a shared MLP to compute channel-wise importance weights.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Ensure hidden plane size is at least 1
        hidden_planes = max(in_planes // ratio, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP implemented as 1x1 Convolutions
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes, in_planes, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply shared MLP to both average and max pooled features
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))

        # Element-wise summation
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for CBAM.

    Uses Mixed Pooling (Average + Max) across the channel dimension to aggregate
    channel information, followed by a convolution to compute spatial importance weights.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        # Padding ensures the spatial dimensions remain unchanged
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate along channel dimension
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # Convolve and activate
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    Sequentially applies Channel Attention and Spatial Attention to refine features.

    Args:
        in_planes (int): Number of input channels.
        ratio (int): Reduction ratio for the Channel Attention MLP. Default: 16.
        kernel_size (int): Kernel size for the Spatial Attention convolution. Default: 7.
    """

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # Refine channels
        out = x * self.ca(x)
        # Refine spatial features
        out = out * self.sa(out)
        return out


class WideConvBlock(nn.Module):
    """
    Wide Convolution Block.

    Standard 3x3 Convolution followed by Batch Normalization and ReLU.
    Designed to maintain a specific output width (default 128) to avoid bottlenecks.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels. Default: 128.
    """

    def __init__(self, in_channels, out_channels=128):
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
