import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for efficient mobile network design.
    Aggregates features along spatial directions to capture long-range dependencies.
    """

    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # Ensure the reduction doesn't make the channel count too small
        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Feature aggregation along two spatial directions
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (N, C, W, 1)

        # Concatenate along the spatial dimension
        y = torch.cat([x_h, x_w], dim=2)

        # Shared 1x1 convolution
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split the features back
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (N, C, 1, W)

        # Generate attention maps
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context.
    """

    def __init__(self, in_channels, out_channels, dilation_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        modules = []

        # 1x1 Convolution
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(),
            )
        )

        # Dilated Convolutions
        for rate in dilation_rates:
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

        # Image Pooling (Global Average Pooling)
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(),
            )
        )

        self.convs = nn.ModuleList(modules)

        # Final projection layer
        # Input channels = out_channels * (number of dilations + 1x1 conv + image pooling)
        total_in_channels = out_channels * (len(dilation_rates) + 2)
        self.project = nn.Sequential(
            nn.Conv2d(total_in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs[:-1]:
            res.append(conv(x))

        # Handle Image Pooling separately to upsample back to input size
        global_feat = self.convs[-1](x)
        global_feat = F.interpolate(
            global_feat, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        res.append(global_feat)

        res = torch.cat(res, dim=1)
        return self.project(res)


class AttentionGate(nn.Module):
    """
    Attention Gate to suppress irrelevant regions in skip connections.
    Uses the gating signal (from decoder) to weight the skip connection (from encoder).
    """

    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Number of channels in the gating signal (decoder feature).
            F_l: Number of channels in the skip connection (encoder feature).
            F_int: Number of intermediate channels.
        """
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

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (decoder), x: skip connection (encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Align dimensions if necessary (e.g., if g is smaller than x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(
                g1, size=x1.shape[2:], mode="bilinear", align_corners=False
            )

        # Additive attention
        psi = self.relu(g1 + x1)

        # Generate attention coefficients
        psi = self.psi(psi)

        # Scale the skip connection
        return x * psi


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with optional Coordinate Attention.
    """

    def __init__(self, in_channels, out_channels, stride=1, use_ca=True):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_ca = use_ca
        if use_ca:
            self.ca = CoordinateAttention(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.act2 = nn.SiLU()

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
