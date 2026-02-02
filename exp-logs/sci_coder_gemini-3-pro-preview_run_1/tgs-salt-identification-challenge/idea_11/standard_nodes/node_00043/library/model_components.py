import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    This module randomly drops the input tensor with probability `drop_prob` during training.
    Used for regularizing deep residual networks.
    """

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        # Compute shape for broadcasting: (batch_size, 1, 1, 1) for 4D input
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        # Generate binary mask
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize

        # Scale output to maintain expected value
        output = x.div(keep_prob) * random_tensor
        return output


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module.
    Factorizes channel attention into two parallel 1D feature encoding operations
    to capture long-range dependencies along one spatial direction while preserving
    precise positional information along the other.
    """

    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()

        # Ensure a reasonable minimum number of channels in the bottleneck
        mid_channels = max(8, in_channels // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(
            mid_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv_w = nn.Conv2d(
            mid_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 1. Coordinate Information Embedding
        x_h = self.pool_h(x)  # (N, C, H, 1)
        x_w = self.pool_w(x)  # (N, C, 1, W)

        # Concatenate feature maps along spatial dimension
        # Permute x_w to (N, C, W, 1) to stack vertically with x_h
        y = torch.cat([x_h, x_w.permute(0, 1, 3, 2)], dim=2)  # (N, C, H+W, 1)

        # 2. Coordinate Attention Generation
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act1(y)

        # Split back into height and width features
        x_h_new, x_w_new = torch.split(y, [h, w], dim=2)

        # Expand channels back to original
        x_w_new = x_w_new.permute(0, 1, 3, 2)  # (N, C, 1, W)

        a_h = self.sigmoid(self.conv_h(x_h_new))
        a_w = self.sigmoid(self.conv_w(x_w_new))

        # Re-weight the input
        out = identity * a_h * a_w
        return out


class ResidualBlock(nn.Module):
    """
    Residual Block with optional Coordinate Attention and DropPath.

    Structure:
    Input -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> [CA] -> [DropPath] + [Shortcut] -> ReLU
    """

    def __init__(
        self, in_channels, out_channels, stride=1, use_ca=False, drop_path_rate=0.0
    ):
        super(ResidualBlock, self).__init__()

        self.use_ca = use_ca
        self.drop_path_rate = drop_path_rate

        # First convolution
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

        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Optional Coordinate Attention
        if self.use_ca:
            self.ca = CoordinateAttention(out_channels)

        # Optional Drop Path (Stochastic Depth)
        if self.drop_path_rate > 0:
            self.drop_path = DropPath(drop_path_rate)

        # Shortcut connection handling
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_ca:
            out = self.ca(out)

        if self.drop_path_rate > 0:
            out = self.drop_path(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out
