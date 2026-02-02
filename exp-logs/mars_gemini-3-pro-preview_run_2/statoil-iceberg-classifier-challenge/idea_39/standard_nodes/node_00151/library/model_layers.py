import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPooling(nn.Module):
    """
    Implements Dual-Stream Pooling by concatenating Max Pooling and Min Pooling outputs.
    This preserves both peak signal intensity (reflections) and shadow features (voids).

    Output channels = 2 * Input channels.
    """

    def __init__(self, kernel_size=2, stride=2):
        super(DualPooling, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # Max pooling captures high intensity signals (peaks)
        x_max = self.pool(x)

        # Min pooling captures low intensity signals (shadows)
        # Implemented as -MaxPool(-x)
        x_min = -self.pool(-x)

        # Concatenate along the channel dimension
        return torch.cat([x_max, x_min], dim=1)


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (part of CBAM).
    Uses Mixed Pooling (Avg + Max) to aggregate spatial information.
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Calculate hidden dimension, ensuring at least 1 neuron
        hidden_channels = max(1, in_channels // reduction_ratio)

        # Shared MLP
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # Average Pooling Path
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out)

        # Max Pooling Path
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out)

        # Element-wise Sum and Sigmoid activation
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (part of CBAM).
    Uses Mixed Pooling (Avg + Max) across channels to generate a spatial attention map.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Padding to maintain spatial dimensions
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise Average Pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Channel-wise Max Pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate features
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # Convolution and Sigmoid activation
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies Channel Attention and Spatial Attention.
    """

    def __init__(self, channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        # Refine features with Channel Attention
        x = x * self.channel_att(x)
        # Refine features with Spatial Attention
        x = x * self.spatial_att(x)
        return x


class WideBlock(nn.Module):
    """
    Wide-Body Delayed-Integration Block.

    Architecture Sequence:
    1. Conv2d (3x3) mapping in_channels -> out_channels
    2. BatchNorm
    3. ReLU
    4. CBAM (Attention Refinement)
    5. DualPooling (Max + Min)

    The output channel dimension will be 2 * out_channels due to DualPooling.
    """

    def __init__(self, in_channels, out_channels):
        super(WideBlock, self).__init__()

        # Wide Convolution
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Pre-Pooling Attention
        self.cbam = CBAM(out_channels)

        # Dual-Stream Pooling
        self.pool = DualPooling(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)
        x = self.pool(x)
        return x
