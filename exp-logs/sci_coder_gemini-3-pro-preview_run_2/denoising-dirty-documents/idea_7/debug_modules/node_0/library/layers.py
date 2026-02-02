import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module.
    Captures long-range dependencies with precise positional information by decomposing
    global pooling into vertical and horizontal directions.
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

        # 1. Feature Encoding
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # N, C, W, 1

        # 2. Concatenation and Shared Transformation
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 3. Split and Excite
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out


class AttentionGate(nn.Module):
    """
    Attention Gate.
    Filters features from the encoder (skip connection) using the context from the decoder (gating signal).
    Suppresses irrelevant regions (noise) in the skip connection.
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.SiLU()

    def forward(self, g, x):
        # g: gating signal (from decoder)
        # x: skip connection (from encoder)

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Align dimensions if necessary (upsample g to match x)
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(
                g1, size=x1.size()[2:], mode="bilinear", align_corners=False
            )

        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)

        return x * alpha


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    Captures multi-scale context using parallel branches with different dilation rates.
    """

    def __init__(self, in_channels, out_channels, dilations=[6, 12, 18]):
        super(ASPP, self).__init__()

        # 1. Global Average Pooling Branch (Image-level features)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        # 2. 1x1 Convolution Branch
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        # 3. Dilated Convolution Branches
        self.dilated_convs = nn.ModuleList()
        for dilation in dilations:
            self.dilated_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(),
                )
            )

        # Projection layer
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        # Image Pooling
        res.append(
            F.interpolate(
                self.global_avg_pool(x),
                size=x.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        # 1x1 Conv
        res.append(self.conv1x1(x))
        # Dilated Convs
        for conv in self.dilated_convs:
            res.append(conv(x))

        res = torch.cat(res, dim=1)
        return self.project(res)


class ResBlock(nn.Module):
    """
    Residual Block with Coordinate Attention.
    Standard ResNet block adapted with SiLU activation and optional Coordinate Attention.
    """

    def __init__(self, in_channels, out_channels, use_ca=True):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_ca = use_ca
        if use_ca:
            self.ca = CoordinateAttention(out_channels)

        self.act2 = nn.SiLU()

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_ca:
            out = self.ca(out)

        out += residual
        out = self.act2(out)
        return out


class DeepSupervisionHead(nn.Module):
    """
    Deep Supervision Head.
    A simple projection layer to generate auxiliary predictions from intermediate decoder features.
    """

    def __init__(self, in_channels, out_channels=1):
        super(DeepSupervisionHead, self).__init__()
        # Simple 1x1 convolution to project features to output space (noise residual)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
