import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module.
    Captures long-range dependencies with precise positional information by
    decomposing channel attention into two 1D feature encoding processes.
    """

    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Pool
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concat
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Expand
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out


class SKFusion(nn.Module):
    """
    Selective Kernel Fusion Module.
    Splits input into Local (3x3) and Context (3x3 dilated) branches,
    and fuses them using adaptive channel-wise attention.
    """

    def __init__(self, in_channels, out_channels, stride=1, reduction=16, groups=1):
        super(SKFusion, self).__init__()

        # Branch 1: Local (3x3, dilation=1)
        self.conv_local = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=1,
            groups=groups,
            bias=False,
        )

        # Branch 2: Context (3x3, dilation=2 -> effectively 5x5 receptive field)
        self.conv_context = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=2,
            dilation=2,
            groups=groups,
            bias=False,
        )

        # Fusion logic
        self.out_channels = out_channels
        d = max(int(out_channels / reduction), 32)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(out_channels, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d),
            nn.SiLU(),
        )
        self.fc_expand = nn.Conv2d(d, 2 * out_channels, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 1. Split
        u1 = self.conv_local(x)
        u2 = self.conv_context(x)

        # 2. Fuse (Sum)
        u = u1 + u2

        # 3. Select (Attention)
        s = self.gap(u)
        z = self.fc(s)
        weights = self.fc_expand(z)

        # Reshape to (B, 2, C, 1, 1) to apply softmax over the 2 branches
        b, c, _, _ = u.shape
        weights = weights.view(b, 2, c, 1, 1)
        weights = self.softmax(weights)

        # 4. Apply weights
        # u1 is (B, C, H, W), weights[:, 0] is (B, C, 1, 1)
        v = (weights[:, 0] * u1) + (weights[:, 1] * u2)

        return v


class CSKBlock(nn.Module):
    """
    Coordinate Selective Kernel Residual Block.
    Structure: Input -> SKFusion -> BN -> SiLU -> CoordinateAttention -> Add Residual
    """

    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(CSKBlock, self).__init__()

        self.sk_fusion = SKFusion(
            in_channels, out_channels, stride=stride, reduction=reduction
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.ca = CoordinateAttention(out_channels, reduction=reduction)

        # Shortcut connection
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.sk_fusion(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.ca(out)

        out += residual
        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    Captures multi-scale global context at the bottleneck.
    """

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        # 1x1 Conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        # 3x3 Conv, rate 6
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        # 3x3 Conv, rate 12
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3, padding=12, dilation=12, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        # 3x3 Conv, rate 18
        self.branch4 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3, padding=18, dilation=18, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        # Global Pooling Branch
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        b5 = self.branch5(x)
        b5 = F.interpolate(b5, size=(h, w), mode="bilinear", align_corners=False)

        out = torch.cat([b1, b2, b3, b4, b5], dim=1)
        out = self.conv_cat(out)

        return out
