import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import BASE_FILTERS, IN_CHANNELS, OUT_CHANNELS


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
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Split pooling
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate and process
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Generate attention maps
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_w * a_h
        return out


class InvertedResidualBlock(nn.Module):
    """
    Inverted Residual Block with Standard Convolutions and Coordinate Attention.
    Structure: Expansion (1x1) -> Standard Spatial (3x3) -> CA -> Projection (1x1)
    """

    def __init__(self, in_channels, out_channels, expansion_factor=2):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(in_channels * expansion_factor)

        # 1. Expansion
        self.expansion = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        )

        # 2. Spatial Filtering (Standard Convolution, NOT depthwise)
        self.spatial = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        )

        # 3. Coordinate Attention (Applied on expanded features)
        self.ca = CoordinateAttention(hidden_dim)

        # 4. Projection
        self.projection = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # 5. Shortcut
        self.use_shortcut = True
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)

        x = self.expansion(x)
        x = self.spatial(x)
        x = self.ca(x)
        x = self.projection(x)

        return x + res


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
                nn.SiLU(),
            )
        )

        # Atrous Convolutions
        dilations = [6, 12, 18]
        for rate in dilations:
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
                    nn.SiLU(),
                )
            )

        # Global Pooling
        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        self.convs = nn.ModuleList(modules)

        # Projection after concatenation
        # 5 branches total (1x1 + 3 atrous + 1 global)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        # Global pooling branch
        gp = self.global_pooling(x)
        gp = F.interpolate(gp, size=x.shape[2:], mode="bilinear", align_corners=False)
        res.append(gp)

        res = torch.cat(res, dim=1)
        return self.project(res)


class ICResUNet(nn.Module):
    """
    Inverted Coordinate ResUNet.
    Predicts the noise residual.
    """

    def __init__(self):
        super(ICResUNet, self).__init__()

        filters = [BASE_FILTERS, BASE_FILTERS * 2, BASE_FILTERS * 4, BASE_FILTERS * 8]

        # --- Encoder ---
        # Initial processing
        self.inc_conv = nn.Conv2d(
            IN_CHANNELS, filters[0], kernel_size=3, padding=1, bias=False
        )
        self.inc_bn = nn.BatchNorm2d(filters[0])
        self.inc_act = nn.SiLU()
        self.inc_block = InvertedResidualBlock(filters[0], filters[0])

        # Downsampling stages
        self.down1_pool = nn.MaxPool2d(2)
        self.down1_block = InvertedResidualBlock(filters[0], filters[1])

        self.down2_pool = nn.MaxPool2d(2)
        self.down2_block = InvertedResidualBlock(filters[1], filters[2])

        self.down3_pool = nn.MaxPool2d(2)
        self.down3_block = InvertedResidualBlock(filters[2], filters[3])

        # --- Bridge ---
        self.aspp = ASPP(filters[3], filters[3])

        # --- Decoder ---
        # Up 1
        self.up1 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.dec1_block = InvertedResidualBlock(
            filters[3], filters[2]
        )  # Input: 256+256=512 -> 256

        # Up 2
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.dec2_block = InvertedResidualBlock(
            filters[2], filters[1]
        )  # Input: 128+128=256 -> 128

        # Up 3
        self.up3 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.dec3_block = InvertedResidualBlock(
            filters[1], filters[0]
        )  # Input: 64+64=128 -> 64

        # --- Output Head ---
        self.out_conv = nn.Conv2d(filters[0], OUT_CHANNELS, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.inc_conv(x)
        x0 = self.inc_bn(x0)
        x0 = self.inc_act(x0)
        x0 = self.inc_block(x0)  # 64

        x1 = self.down1_pool(x0)
        x1 = self.down1_block(x1)  # 128

        x2 = self.down2_pool(x1)
        x2 = self.down2_block(x2)  # 256

        x3 = self.down3_pool(x2)
        x3 = self.down3_block(x3)  # 512

        # Bridge
        b = self.aspp(x3)  # 512

        # Decoder
        d1 = self.up1(b)  # 256
        # Handle potential padding issues if dimensions are not perfectly divisible
        if d1.size() != x2.size():
            d1 = F.interpolate(
                d1, size=x2.shape[2:], mode="bilinear", align_corners=False
            )
        d1 = torch.cat([x2, d1], dim=1)  # 256 + 256 = 512
        d1 = self.dec1_block(d1)  # 256

        d2 = self.up2(d1)  # 128
        if d2.size() != x1.size():
            d2 = F.interpolate(
                d2, size=x1.shape[2:], mode="bilinear", align_corners=False
            )
        d2 = torch.cat([x1, d2], dim=1)  # 128 + 128 = 256
        d2 = self.dec2_block(d2)  # 128

        d3 = self.up3(d2)  # 64
        if d3.size() != x0.size():
            d3 = F.interpolate(
                d3, size=x0.shape[2:], mode="bilinear", align_corners=False
            )
        d3 = torch.cat([x0, d3], dim=1)  # 64 + 64 = 128
        d3 = self.dec3_block(d3)  # 64

        # Output (Noise Prediction)
        out = self.out_conv(d3)

        return out
