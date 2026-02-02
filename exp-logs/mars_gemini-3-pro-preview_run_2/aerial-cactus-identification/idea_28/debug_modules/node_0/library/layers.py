import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import library.utils as utils


class BlurPool(nn.Module):
    """
    Anti-Aliased Downsampling using a fixed low-pass filter.
    Reference: 'Making Convolutional Networks Shift-Invariant Again', ICML 2019.
    """

    def __init__(self, channels, stride=1):
        super(BlurPool, self).__init__()
        self.channels = channels
        self.stride = stride
        # Create a [1, 2, 1] derived 3x3 smoothing kernel
        # Outer product of [1, 2, 1] is [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        kernel = kernel / 16.0  # Normalize
        kernel = kernel.view(1, 1, 3, 3)
        kernel = kernel.repeat(channels, 1, 1, 1)

        # Register as a buffer so it's part of the state_dict but not a learnable parameter
        self.register_buffer("kernel", kernel)

        # Reflection padding to reduce boundary artifacts
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x):
        # If stride is 1, strictly speaking we don't need to blur if we aren't downsampling,
        # but consistent application enforces shift invariance.
        # However, for efficiency in this block design, we typically use it for stride > 1.
        if self.stride == 1:
            return x

        x = self.pad(x)
        # Depthwise convolution (groups=channels)
        return F.conv2d(x, self.kernel, stride=self.stride, groups=self.channels)


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    Computes f = (mean(x^p))^(1/p).
    p -> 1: Average Pooling
    p -> infinity: Max Pooling
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W) -> (B, C, 1, 1)
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid NaN in pow()
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for Efficient Mobile Network Design.
    Captures long-range dependencies with precise positional information.
    """

    def __init__(self, inp, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Pool spatially
        x_h = self.pool_h(x)  # (N, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (N, C, 1, W) -> (N, C, W, 1)

        # Concatenate along spatial dimension to process together
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Generate attention maps
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


class Res2NeXtBlock(nn.Module):
    """
    Custom Wide Anti-Aliased Coordinate-Res2NeXt Block.

    Args:
        in_planes (int): Input channel count.
        planes (int): Base channel count (used for bottleneck width calculation).
        stride (int): Stride for downsampling.
        cardinality (int): Number of groups for ResNeXt convolution.
        base_width (int): Base width for ResNeXt width calculation.
        scales (int): Number of scales for Res2Net hierarchical split.
        expansion (int): Expansion factor for the output 1x1 convolution.
    """

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        cardinality=32,
        base_width=4,
        scales=4,
        expansion=4,
    ):
        super(Res2NeXtBlock, self).__init__()

        # Calculate bottleneck width based on ResNeXt formula
        # width = floor(planes * (base_width/64)) * cardinality
        # For small images/wide networks, we ensure width is sufficient.
        width = math.floor(planes * (base_width / 64.0)) * cardinality
        width = int(max(width, cardinality))

        self.scales = scales
        self.stride = stride

        # Ensure width is divisible by scales for Res2Net splitting
        if width % scales != 0:
            width = int(math.ceil(width / scales) * scales)

        self.width = width
        self.width_per_scale = width // scales

        # Calculate groups per scale to maintain total cardinality
        # If cardinality=32 and scales=4, each scale conv has groups=8.
        self.groups_per_scale = cardinality // scales
        if self.groups_per_scale == 0:
            self.groups_per_scale = 1

        # 1x1 Compression
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        # Res2Net Hierarchical Convolutions (3x3)
        # We need (scales - 1) convolutions. The first scale is identity/passed through.
        self.convs = nn.ModuleList()
        for i in range(scales - 1):
            self.convs.append(
                nn.Conv2d(
                    self.width_per_scale,
                    self.width_per_scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=self.groups_per_scale,
                    bias=False,
                )
            )
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(self.width_per_scale) for _ in range(scales - 1)]
        )

        # BlurPool for downsampling (applied after 3x3s if stride > 1)
        if stride > 1:
            self.blur = BlurPool(width, stride=stride)
        else:
            self.blur = nn.Identity()

        # 1x1 Expansion
        self.conv3 = nn.Conv2d(width, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)

        # Coordinate Attention
        self.ca = CoordinateAttention(planes * expansion)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        self.downsample = None
        if stride != 1 or in_planes != planes * expansion:
            layers = []
            if stride > 1:
                # Use BlurPool for anti-aliased downsampling in shortcut
                layers.append(BlurPool(in_planes, stride=stride))
                layers.append(
                    nn.Conv2d(in_planes, planes * expansion, kernel_size=1, bias=False)
                )
            else:
                layers.append(
                    nn.Conv2d(in_planes, planes * expansion, kernel_size=1, bias=False)
                )

            layers.append(nn.BatchNorm2d(planes * expansion))
            self.downsample = nn.Sequential(*layers)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # --- Res2Net Processing ---
        # Split feature map into 'scales' chunks
        xs = torch.split(out, self.width_per_scale, dim=1)
        ys = []

        # First chunk
        # In standard Res2Net, y1 = x1
        y = xs[0]
        ys.append(y)

        # Subsequent chunks
        for i in range(1, self.scales):
            # y_i = Conv(x_i + y_{i-1})
            # Note: ys[-1] is the output of the previous iteration
            if i == 1:
                y = self.convs[i - 1](xs[i])
            else:
                y = self.convs[i - 1](xs[i] + ys[-1])
            y = self.relu(self.bns[i - 1](y))
            ys.append(y)

        out = torch.cat(ys, dim=1)
        # --------------------------

        # Apply anti-aliased downsampling if needed
        out = self.blur(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply Coordinate Attention
        out = self.ca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
