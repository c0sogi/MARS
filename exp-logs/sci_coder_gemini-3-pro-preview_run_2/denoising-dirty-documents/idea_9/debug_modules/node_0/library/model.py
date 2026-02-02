import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for efficient mobile network design.
    Aggregates features along H and W directions to capture long-range dependencies.
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

        # Pool along H and W
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

        # Attention maps
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out


class Res2NetBlock(nn.Module):
    """
    Res2Net Block: A New Multi-scale Backbone Architecture.
    Constructs hierarchical residual-like connections within a single residual block.
    """

    def __init__(self, in_channels, out_channels, scale=4, stride=1):
        super(Res2NetBlock, self).__init__()

        self.scale = scale
        self.width = out_channels // scale
        self.stride = stride

        # Shortcut connection for dimension matching
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        # 1x1 Conv to reduce/expand dimensions before splitting
        self.conv1 = nn.Conv2d(
            in_channels, self.width * scale, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.width * scale)

        # Hierarchical 3x3 convolutions
        self.nums = scale - 1
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    self.width,
                    self.width,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                )
                for _ in range(self.nums)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm2d(self.width) for _ in range(self.nums)])

        # 1x1 Conv to fuse features
        self.conv3 = nn.Conv2d(
            self.width * scale, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.act = nn.SiLU()
        self.ca = CoordinateAttention(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        # Split features into groups
        spx = torch.split(out, self.width, 1)

        y = spx[
            0
        ]  # The first group goes through directly (or processed if stride>1, but here stride=1)
        sp = y
        output_splits = []
        output_splits.append(y)

        # Hierarchical processing
        for i in range(self.nums):
            if i == 0:
                sp = spx[i + 1]
            else:
                sp = sp + spx[i + 1]  # Add output of previous conv to current input

            sp = self.convs[i](sp)
            sp = self.bns[i](sp)
            sp = self.act(sp)
            output_splits.append(sp)

        out = torch.cat(output_splits, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply Coordinate Attention
        out = self.ca(out)

        # Residual connection
        out += self.shortcut(x)
        out = self.act(out)

        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    Captures multi-scale context using parallel dilated convolutions.
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

        # Dilated Convolutions
        rates = [6, 12, 18]
        for rate in rates:
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

        # Global Average Pooling
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(),
            )
        )

        self.convs = nn.ModuleList(modules)

        # Project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            out = conv(x)
            # Upsample global pooling branch
            if out.size(2) != x.size(2):
                out = F.interpolate(
                    out, size=x.shape[2:], mode="bilinear", align_corners=False
                )
            res.append(out)

        res = torch.cat(res, dim=1)
        return self.project(res)


class CoRes2NetUNet(nn.Module):
    """
    Coordinate Res2Net U-Net (CoRes2Net-UNet).
    Integrates Res2Net blocks, Coordinate Attention, and ASPP into a U-Net architecture
    for robust image denoising via global residual learning.
    """

    def __init__(self, in_channels=1, out_channels=1, base_filters=64):
        super(CoRes2NetUNet, self).__init__()

        # Initial Convolution
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.SiLU(),
        )

        # Encoder (Downsampling)
        self.enc1 = Res2NetBlock(base_filters, base_filters)
        self.down1 = nn.MaxPool2d(2)

        self.enc2 = Res2NetBlock(base_filters, base_filters * 2)
        self.down2 = nn.MaxPool2d(2)

        self.enc3 = Res2NetBlock(base_filters * 2, base_filters * 4)
        self.down3 = nn.MaxPool2d(2)

        # Bottleneck
        self.aspp = ASPP(base_filters * 4, base_filters * 8)

        # Decoder (Upsampling)
        self.up3 = nn.ConvTranspose2d(
            base_filters * 8, base_filters * 4, kernel_size=2, stride=2
        )
        self.dec3 = Res2NetBlock(
            base_filters * 8, base_filters * 4
        )  # Concat: 4+4 -> 8 in, 4 out

        self.up2 = nn.ConvTranspose2d(
            base_filters * 4, base_filters * 2, kernel_size=2, stride=2
        )
        self.dec2 = Res2NetBlock(
            base_filters * 4, base_filters * 2
        )  # Concat: 2+2 -> 4 in, 2 out

        self.up1 = nn.ConvTranspose2d(
            base_filters * 2, base_filters, kernel_size=2, stride=2
        )
        self.dec1 = Res2NetBlock(
            base_filters * 2, base_filters
        )  # Concat: 1+1 -> 2 in, 1 out

        # Output Layer
        self.outc = nn.Conv2d(base_filters, out_channels, 1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x1 = self.enc1(x1)

        x2 = self.down1(x1)
        x2 = self.enc2(x2)

        x3 = self.down2(x2)
        x3 = self.enc3(x3)

        # Bottleneck
        x_mid = self.down3(x3)
        x_mid = self.aspp(x_mid)

        # Decoder
        d3 = self.up3(x_mid)
        # Handle padding issues if dimensions are not perfect powers of 2
        if d3.size(2) != x3.size(2) or d3.size(3) != x3.size(3):
            d3 = F.interpolate(
                d3, size=x3.shape[2:], mode="bilinear", align_corners=False
            )
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.size(2) != x2.size(2) or d2.size(3) != x2.size(3):
            d2 = F.interpolate(
                d2, size=x2.shape[2:], mode="bilinear", align_corners=False
            )
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.size(2) != x1.size(2) or d1.size(3) != x1.size(3):
            d1 = F.interpolate(
                d1, size=x1.shape[2:], mode="bilinear", align_corners=False
            )
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.dec1(d1)

        # Predict Noise Residual
        noise_pred = self.outc(d1)

        # Global Residual Learning: Return Clean = Input - Noise
        return x - noise_pred
