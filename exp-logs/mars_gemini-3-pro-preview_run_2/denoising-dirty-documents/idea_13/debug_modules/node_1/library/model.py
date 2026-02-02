import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for efficient mobile network design.
    Aggregates features along horizontal and vertical directions to capture
    long-range dependencies and precise positional information.
    """

    def __init__(self, inp, reduction=32):
        super().__init__()
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

        # Pool
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate and process
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Attention maps
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out


class Res2NetBlock(nn.Module):
    """
    Res2Net Block: Hierarchical multi-scale feature extraction.
    Splits features into groups and connects them in a residual cascade.
    """

    def __init__(self, in_channels, out_channels, scale=4, stride=1):
        super().__init__()
        self.scale = scale
        # Ensure width is divisible
        self.width = out_channels // scale
        self.stride = stride

        # Residual path downsampling if dimensions change
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        # 1x1 expansion/projection
        self.conv1 = nn.Conv2d(
            in_channels, self.width * scale, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.width * scale)

        # Hierarchical 3x3 convolutions
        # We need (scale - 1) convolutions. The first group is passed through or added.
        self.nums = scale
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    self.width,
                    self.width,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
                for _ in range(self.scale - 1)
            ]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(self.width) for _ in range(self.scale - 1)]
        )

        # 1x1 fusion
        self.conv3 = nn.Conv2d(
            self.width * scale, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

        # Coordinate Attention
        self.ca = CoordinateAttention(out_channels)

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        # Split into groups
        spx = torch.split(out, self.width, 1)

        # First group (y1 = x1)
        y = spx[0]
        out_list = [y]

        # Cascade: yi = Conv(xi + y_{i-1})
        for i in range(self.nums - 1):
            sp_curr = spx[i + 1] + y
            y = self.convs[i](sp_curr)
            y = self.bns[i](y)
            y = self.act(y)
            out_list.append(y)

        out = torch.cat(out_list, 1)
        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out)

        out += residual
        out = self.act(out)
        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Captures multi-scale context using dilated convolutions.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Conservative dilations for 128x128 patch inputs (bottleneck is 8x8)
        dilations = [1, 2, 4]

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilations[0],
                dilation=dilations[0],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilations[1],
                dilation=dilations[1],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilations[2],
                dilation=dilations[2],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        return x


class CoRes2NetUNet(nn.Module):
    """
    Coordinate Res2Net U-Net.
    Combines Res2Net blocks, Coordinate Attention, and ASPP in a U-Net architecture.
    Predicts the noise residual.
    """

    def __init__(self):
        super().__init__()
        base = Config.BASE_FILTERS
        scale = Config.RES2NET_SCALE

        # --- Encoder ---
        # Initial Conv
        self.inc = nn.Sequential(
            nn.Conv2d(1, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.SiLU(),
        )

        # Stage 1
        self.down1 = nn.MaxPool2d(2)
        self.res1 = Res2NetBlock(base, base * 2, scale=scale)

        # Stage 2
        self.down2 = nn.MaxPool2d(2)
        self.res2 = Res2NetBlock(base * 2, base * 4, scale=scale)

        # Stage 3
        self.down3 = nn.MaxPool2d(2)
        self.res3 = Res2NetBlock(base * 4, base * 8, scale=scale)

        # Stage 4
        self.down4 = nn.MaxPool2d(2)
        self.res4 = Res2NetBlock(base * 8, base * 16, scale=scale)

        # --- Bottleneck ---
        self.aspp = ASPP(base * 16, base * 16)

        # --- Decoder ---
        # Up 1
        self.up1 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec_res1 = Res2NetBlock(
            base * 16, base * 8, scale=scale
        )  # Input: cat(up, skip) = 8+8=16

        # Up 2
        self.up2 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec_res2 = Res2NetBlock(base * 8, base * 4, scale=scale)  # Input: 4+4=8

        # Up 3
        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec_res3 = Res2NetBlock(base * 4, base * 2, scale=scale)  # Input: 2+2=4

        # Up 4
        self.up4 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec_res4 = Res2NetBlock(base * 2, base, scale=scale)  # Input: 1+1=2

        # --- Output Head ---
        self.outc = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)  # [B, 64, H, W]

        x2 = self.down1(x1)  # [B, 64, H/2, W/2]
        x2 = self.res1(x2)  # [B, 128, H/2, W/2]

        x3 = self.down2(x2)  # [B, 128, H/4, W/4]
        x3 = self.res2(x3)  # [B, 256, H/4, W/4]

        x4 = self.down3(x3)  # [B, 256, H/8, W/8]
        x4 = self.res3(x4)  # [B, 512, H/8, W/8]

        x5 = self.down4(x4)  # [B, 512, H/16, W/16]
        x5 = self.res4(x5)  # [B, 1024, H/16, W/16]

        # Bottleneck
        b = self.aspp(x5)  # [B, 1024, H/16, W/16]

        # Decoder
        u1 = self.up1(b)  # [B, 512, H/8, W/8]
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.dec_res1(u1)  # [B, 512, H/8, W/8]

        u2 = self.up2(u1)  # [B, 256, H/4, W/4]
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.dec_res2(u2)  # [B, 256, H/4, W/4]

        u3 = self.up3(u2)  # [B, 128, H/2, W/2]
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.dec_res3(u3)  # [B, 128, H/2, W/2]

        u4 = self.up4(u3)  # [B, 64, H, W]
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.dec_res4(u4)  # [B, 64, H, W]

        # Prediction
        noise_pred = self.outc(u4)

        return noise_pred
