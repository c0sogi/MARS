import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: 3x3 Convolution -> BatchNorm -> ReLU.
    Maintains spatial resolution (padding=1).
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
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


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling: Concatenates Max Pooling (Peaks) and Min Pooling (Shadows).

    Mechanism:
    - Max Pooling captures the high-intensity radar backscatter (icebergs/ships).
    - Min Pooling captures the low-intensity shadows, which are critical for shape analysis.

    Output:
    - Concatenates the two pooled maps along the channel dimension.
    - Output channels = 2 * Input channels.
    """

    def __init__(self):
        super(DualPooling, self).__init__()
        # Standard 2x2 pooling with stride 2
        self.pool_size = 2
        self.stride = 2

    def forward(self, x):
        # Max Pooling (Peaks)
        max_p = F.max_pool2d(x, kernel_size=self.pool_size, stride=self.stride)

        # Min Pooling (Shadows)
        # PyTorch does not have a native min_pool2d, so we use -max_pool2d(-x)
        min_p = -F.max_pool2d(-x, kernel_size=self.pool_size, stride=self.stride)

        # Concatenate along channel dimension
        return torch.cat([max_p, min_p], dim=1)


class ChannelAttention(nn.Module):
    """
    Channel Attention Sub-module for CBAM.
    Uses Mixed Pooling (Avg + Max) to aggregate spatial information.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP to produce channel weights
        # Ensure hidden dimension is at least 4 to prevent information collapse
        hidden_planes = max(in_planes // ratio, 4)

        # Using 1x1 Convs as shared MLP for efficiency
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Path 1: Average Pooling
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # Path 2: Max Pooling
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # Sum and Sigmoid
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Sub-module for CBAM.
    Uses Mixed Pooling (Avg + Max) along the channel axis.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Spatial Attention kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        # Convolution to produce spatial map from pooled features
        # Input channels = 2 (1 for Avg, 1 for Max)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average along channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Max along channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # Convolve and Sigmoid
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Refines features sequentially via Channel Attention then Spatial Attention.
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


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Performs Global Channel Recalibration.

    Structure:
    Global Avg Pool -> Dense (Reduction) -> ReLU -> Dense (Expansion) -> Sigmoid -> Scale
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Ensure bottleneck has at least 1 neuron
        hidden_dim = max(channels // reduction, 1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Recalibration
        return x * y.expand_as(x)
