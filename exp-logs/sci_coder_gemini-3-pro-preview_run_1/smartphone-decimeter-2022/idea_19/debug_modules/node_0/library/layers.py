import torch
import torch.nn as nn
import torch.nn.functional as F


class CyclicConv2d(nn.Module):
    """
    2D Convolution with Circular Padding along the Azimuth (Width) dimension
    and Zero Padding along the Time (Height) dimension.

    This layer treats the input as a cylinder: continuous along the width,
    but bounded along the height.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(CyclicConv2d, self).__init__()

        # Standardize kernel_size to tuple (kh, kw)
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # Standardize stride to tuple (sh, sw)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        # Standardize padding to tuple (ph, pw)
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        self.dilation = dilation
        self.groups = groups

        # Separate padding for Height (Time) and Width (Azimuth)
        self.pad_h, self.pad_w = self.padding

        # Initialize the underlying Conv2d.
        # We set width padding to 0 here because we will handle it manually
        # with circular padding in the forward pass.
        # Height padding (Time) is handled normally (Zero Padding) by Conv2d.
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            self.kernel_size,
            self.stride,
            padding=(self.pad_h, 0),
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Time, Azimuth)

        # Apply circular padding to Azimuth (Width dimension)
        # F.pad format for 4D input: (pad_left, pad_right, pad_top, pad_bottom)
        if self.pad_w > 0:
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode="circular")

        # Apply convolution (Height padding is handled internally)
        return self.conv(x)


class ResidualBlock2D(nn.Module):
    """
    A ResNet-style residual block using CyclicConv2d layers.

    Structure:
    Input -> CyclicConv3x3 -> BN -> ReLU -> CyclicConv3x3 -> BN -> Add -> ReLU
         |                                                       ^
         --------------------(Projection if needed)---------------
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock2D, self).__init__()

        # First convolution: handles stride and channel change
        self.conv1 = CyclicConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolution: stride 1, keeps dimensions
        self.conv2 = CyclicConv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection handling
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            # Projection shortcut
            # 1x1 convolution doesn't strictly need circular padding (padding=0),
            # but we use CyclicConv2d class for consistency.
            self.downsample = nn.Sequential(
                CyclicConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
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
