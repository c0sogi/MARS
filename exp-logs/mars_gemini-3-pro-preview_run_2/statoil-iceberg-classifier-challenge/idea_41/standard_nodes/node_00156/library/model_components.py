import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) part of CBAM.
    Utilizes Mixed Pooling (Average + Max) to aggregate spatial information.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Ensure hidden planes is at least 1
        hidden_planes = max(1, in_planes // ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP implemented as 1x1 Convolutions to avoid reshaping
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_planes, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate descriptors
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))

        # Fusion (Sum)
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) part of CBAM.
    Utilizes Mixed Pooling (Average + Max) along the channel axis.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = kernel_size // 2

        # Input channels is 2 because we concat avg and max pool maps
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # Convolution and Activation
        x_out = self.conv1(x_cat)
        return self.sigmoid(x_out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies Channel Attention then Spatial Attention.
    """

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # Refine channels
        out = x * self.ca(x)
        # Refine spatial
        out = out * self.sa(out)
        return out


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling Layer.
    Performs Max Pooling (to capture peaks) and Min Pooling (to capture shadows/water).
    Concatenates the results, doubling the channel depth.
    """

    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(DualPooling, self).__init__()
        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        # Max Pooling (Peaks)
        max_p = self.pool(x)

        # Min Pooling (Shadows)
        # Mathematically: min(x) = -max(-x)
        min_p = -self.pool(-x)

        # Concatenate along channel dimension
        return torch.cat([max_p, min_p], dim=1)


class ConvBnRelu(nn.Module):
    """
    Standard building block: Conv2d -> BatchNorm2d -> ReLU.
    Used for the 'Wide Convolution' blocks in the backbone.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
