import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module.
    Captures long-range dependencies with precise positional information.
    """

    def __init__(self, inp, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)  # Using ReLU as a standard activation

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out


class ResBlock(nn.Module):
    """
    Residual Block with Coordinate Attention.
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> CA -> Add -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.ca = CoordinateAttention(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out)

        out += residual
        out = self.relu(out)
        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    Captures multi-scale context.
    """

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        modules = []

        # 1x1 Conv
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Atrous Convolutions
        dilations = [6, 12, 18]
        for dilation in dilations:
            modules.append(
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
                    nn.ReLU(inplace=True),
                )
            )

        # Global Average Pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.convs = nn.ModuleList(modules)

        # Project back to out_channels
        # 5 branches: 1x1, 3x3 (d=6), 3x3 (d=12), 3x3 (d=18), GAP
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        gap = self.global_avg_pool(x)
        gap = F.interpolate(
            gap, size=x.size()[2:], mode="bilinear", align_corners=False
        )
        res.append(gap)

        res = torch.cat(res, dim=1)
        return self.project(res)


class UNetStage(nn.Module):
    """
    A single U-Net stage with Coordinate ResBlocks and ASPP bottleneck.
    """

    def __init__(self, in_channels, out_channels, base_filters=64):
        super(UNetStage, self).__init__()

        # Initial Convolution
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.e1 = ResBlock(base_filters, base_filters)
        self.down1 = nn.MaxPool2d(2)

        self.e2 = ResBlock(base_filters, base_filters * 2)
        self.down2 = nn.MaxPool2d(2)

        self.e3 = ResBlock(base_filters * 2, base_filters * 4)
        self.down3 = nn.MaxPool2d(2)

        self.e4 = ResBlock(base_filters * 4, base_filters * 8)
        # No downsampling before bottleneck in this design, e4 is the deepest encoder block

        # Bottleneck
        self.aspp = ASPP(base_filters * 8, base_filters * 8)

        # Decoder
        # Up 1
        self.up1 = nn.ConvTranspose2d(
            base_filters * 8, base_filters * 4, kernel_size=2, stride=2
        )
        self.d1 = ResBlock(
            base_filters * 8, base_filters * 4
        )  # Input is concat(up1, e3) = 4+4 = 8

        # Up 2
        self.up2 = nn.ConvTranspose2d(
            base_filters * 4, base_filters * 2, kernel_size=2, stride=2
        )
        self.d2 = ResBlock(
            base_filters * 4, base_filters * 2
        )  # Input is concat(up2, e2) = 2+2 = 4

        # Up 3
        self.up3 = nn.ConvTranspose2d(
            base_filters * 2, base_filters, kernel_size=2, stride=2
        )
        self.d3 = ResBlock(
            base_filters * 2, base_filters
        )  # Input is concat(up3, e1) = 1+1 = 2

        # Output
        self.outc = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x1 = self.e1(x1)  # 64

        x2 = self.down1(x1)
        x2 = self.e2(x2)  # 128

        x3 = self.down2(x2)
        x3 = self.e3(x3)  # 256

        x4 = self.down3(x3)
        x4 = self.e4(x4)  # 512

        # Bottleneck
        b = self.aspp(x4)  # 512

        # Decoder
        d1 = self.up1(b)  # 256
        # Handle potential padding issues if dimensions are not perfectly divisible
        if d1.size()[2:] != x3.size()[2:]:
            d1 = F.interpolate(
                d1, size=x3.size()[2:], mode="bilinear", align_corners=False
            )
        d1 = torch.cat([x3, d1], dim=1)  # 256 + 256 = 512
        d1 = self.d1(d1)  # 256

        d2 = self.up2(d1)  # 128
        if d2.size()[2:] != x2.size()[2:]:
            d2 = F.interpolate(
                d2, size=x2.size()[2:], mode="bilinear", align_corners=False
            )
        d2 = torch.cat([x2, d2], dim=1)  # 128 + 128 = 256
        d2 = self.d2(d2)  # 128

        d3 = self.up3(d2)  # 64
        if d3.size()[2:] != x1.size()[2:]:
            d3 = F.interpolate(
                d3, size=x1.size()[2:], mode="bilinear", align_corners=False
            )
        d3 = torch.cat([x1, d3], dim=1)  # 64 + 64 = 128
        d3 = self.d3(d3)  # 64

        out = self.outc(d3)
        return out


class TSPCResUNet(nn.Module):
    """
    Two-Stage Progressive Coordinate ResUNet.
    Stage 1: Coarse Estimation.
    Stage 2: Fine Refinement.
    """

    def __init__(self):
        super(TSPCResUNet, self).__init__()

        # Stage 1: Input -> Residual
        self.stage1 = UNetStage(
            in_channels=Config.IN_CHANNELS,
            out_channels=Config.OUT_CHANNELS,
            base_filters=Config.BASE_FILTERS,
        )

        # Stage 2: Input (Original + Stage 1 Output) -> Residual
        # Input channels = 1 (Original) + 1 (Stage 1 Pred) = 2
        self.stage2 = UNetStage(
            in_channels=Config.IN_CHANNELS + Config.OUT_CHANNELS,
            out_channels=Config.OUT_CHANNELS,
            base_filters=Config.BASE_FILTERS,
        )

    def forward(self, x):
        # Stage 1
        res1 = self.stage1(x)

        # Stage 2
        # Concatenate original input and stage 1 prediction
        # Detaching res1 is a design choice:
        # If we want gradients to flow through stage 1 from stage 2 loss, we do NOT detach.
        # The prompt implies "Sum of Losses", suggesting we want both stages to learn.
        # Usually, end-to-end training allows gradients to flow back.
        inp2 = torch.cat([x, res1], dim=1)
        res2 = self.stage2(inp2)

        return res1, res2
