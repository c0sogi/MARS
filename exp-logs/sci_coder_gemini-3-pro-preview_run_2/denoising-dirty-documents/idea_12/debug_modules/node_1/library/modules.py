import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with two 3x3 convolutions, BN, and SiLU activation.
    Includes a skip connection with optional 1x1 conv for dimension matching.
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

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
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.act(out)
        return out


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module.
    Pools features along H and W directions to capture long-range dependencies
    and generates attention maps to filter features.
    """

    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Pool
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (N, C, W, 1)

        # Concat along spatial dimension
        y = torch.cat([x_h, x_w], dim=2)  # (N, C, H+W, 1)

        # Transform
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (N, C, 1, W)

        # Generate attention maps
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    Captures multi-scale context using parallel dilated convolutions.
    """

    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        modules = []

        # 1x1 Convolution
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )
        )

        # Atrous (Dilated) Convolutions
        for rate in rates[1:]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
            )

        # Global Average Pooling Branch
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )
        )

        self.convs = nn.ModuleList(modules)

        # Project concatenated outputs
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs[:-1]:
            res.append(conv(x))

        # Global pooling branch requires upsampling to match spatial dims
        global_feat = self.convs[-1](x)
        global_feat = F.interpolate(
            global_feat, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        res.append(global_feat)

        res = torch.cat(res, dim=1)
        return self.project(res)


class SubPixelUpsample(nn.Module):
    """
    Sub-Pixel Upsampling Module.
    Uses Conv + PixelShuffle to upsample feature maps while preserving gradients.
    """

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(SubPixelUpsample, self).__init__()
        # Expand channels to out_channels * scale_factor^2
        self.conv = nn.Conv2d(
            in_channels, out_channels * (scale_factor**2), kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x
