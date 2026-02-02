import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) part of CBAM.
    Aggregates channel information using both Average and Max pooling.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Ensure hidden dimension is at least 1
        hidden_planes = max(1, in_planes // ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) part of CBAM.
    Aggregates spatial information using channel-wise Average and Max pooling.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate along channel dimension
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # Convolution and Sigmoid
        x_out = self.conv1(x_cat)
        return self.sigmoid(x_out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies Channel Attention and Spatial Attention.
    """

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling Layer.
    Performs Max Pooling to capture signal peaks and Min Pooling to capture shadows/troughs.
    The outputs are concatenated along the channel dimension, effectively doubling the channel depth.
    """

    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(DualPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        # We use a separate MaxPool instance logic for MinPool for clarity,
        # though functionally they could share the module if stateless.
        self.pool_op = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        # Max Pooling (Peaks)
        out_max = self.max_pool(x)

        # Min Pooling (Shadows): -MaxPool(-x)
        # This finds the minimum values in the window
        out_min = -self.pool_op(-x)

        # Concatenate along the channel dimension
        # Input: (N, C, H, W) -> Output: (N, 2*C, H_out, W_out)
        return torch.cat([out_max, out_min], dim=1)
