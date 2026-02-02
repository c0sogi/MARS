import torch
import torch.nn as nn
import torch.nn.functional as F


class DualStreamPooling(nn.Module):
    """
    Implements Dual-Stream Pooling: Max Pooling and Min Pooling.
    Cite solution_lesson_node_00173: "Avoid mixing Average Pooling with Max/Min pooling... Dual-Stream Pooling is highly effective".

    Structure:
    - Stream 1: Max Pooling (Captures Signal Peaks)
    - Stream 2: Min Pooling (Captures Signal Shadows) -> Implemented as -MaxPool(-x)

    Output:
    - Concatenation of streams along the channel dimension.
    - Output channels = Input channels * 2
    """

    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(DualStreamPooling, self).__init__()
        self.pool_size = kernel_size
        self.stride = stride
        self.padding = padding

        # We use standard pooling layers
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        # Stream 1: Max Pooling
        out_max = self.max_pool(x)

        # Stream 2: Min Pooling
        # MinPool(x) is equivalent to -MaxPool(-x)
        out_min = -self.max_pool(-x)

        # Concatenate along channel dimension (dim=1)
        return torch.cat([out_max, out_min], dim=1)


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM.
    Uses Mixed Pooling (Max + Avg) as per instructions.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        # Shared MLP
        # To save parameters and compute, we reduce channels by ratio
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes, bias=False),
        )

    def forward(self, x):
        # Global Average Pooling
        avg_out = torch.mean(x, dim=(2, 3))  # (N, C)
        avg_out = self.mlp(avg_out)

        # Global Max Pooling
        max_out = torch.amax(x, dim=(2, 3))  # (N, C)
        max_out = self.mlp(max_out)

        # Sum and Sigmoid
        out = avg_out + max_out
        return torch.sigmoid(out).unsqueeze(2).unsqueeze(3)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for CBAM.
    Uses Mixed Pooling (Max + Avg) across channels.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        # Convolution to collapse the 2 channels (Avg+Max) to 1 attention map
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # Average Pooling across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Max Pooling across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # Conv + Sigmoid
        out = self.conv1(x_cat)
        return torch.sigmoid(out)


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
        # Apply Channel Attention
        out = x * self.ca(x)
        # Apply Spatial Attention
        out = out * self.sa(out)
        return out


class WideBlock(nn.Module):
    """
    WideBlock implementing the 'Delayed-Integration' and 'Sustained Width' strategies.

    Flow:
    1. Wide Convolution (In -> 128 filters)
    2. BatchNorm
    3. ReLU
    4. CBAM (Pre-Pooling Attention)
    5. DualStreamPooling (128 -> 256 filters)
    """

    def __init__(self, in_channels, out_filters=128):
        super(WideBlock, self).__init__()

        # 1. Wide Convolution
        # Maps input channels to fixed width (128)
        self.conv = nn.Conv2d(
            in_channels, out_filters, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_filters)
        self.relu = nn.ReLU(inplace=True)

        # 2. CBAM
        # Applied on the 128-channel feature map
        self.cbam = CBAM(out_filters)

        # 3. DualStreamPooling
        # Expands 128 channels to 256 (128*2)
        self.pool = DualStreamPooling(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # Apply Attention before pooling
        x = self.cbam(x)

        # Apply Dual-Stream Pooling
        x = self.pool(x)
        return x
