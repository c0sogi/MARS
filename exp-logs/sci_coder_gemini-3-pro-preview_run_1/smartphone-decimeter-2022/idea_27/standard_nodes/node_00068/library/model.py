import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with Batch Normalization and ReLU.
    Structure: Conv1D -> BN -> ReLU -> Conv1D -> BN -> Residual Add -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection to match dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (1D).
    Captures multi-scale context using varying dilation rates.
    """

    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12, 18]):
        super(ASPP, self).__init__()

        self.branches = nn.ModuleList()

        # Branch 1: 1x1 Conv
        self.branches.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Branches 2-4: 3x3 Conv with dilation
        for d in dilations[1:]:
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels, out_channels, 3, padding=d, dilation=d, bias=False
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Branch 5: Global Average Pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

        # Final projection
        self.project = nn.Sequential(
            nn.Conv1d(out_channels * (len(dilations) + 1), out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for branch in self.branches:
            res.append(branch(x))

        # Global pooling branch
        gap = self.global_avg_pool(x)
        gap = F.interpolate(gap, size=x.size(2), mode="linear", align_corners=True)
        res.append(gap)

        res = torch.cat(res, dim=1)
        return self.project(res)


class AttentionGate(nn.Module):
    """
    Attention Gate for 1D U-Net.
    Filters features from the skip connection (x) using the gating signal (g) from the decoder.
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: Gating signal (from decoder, upsampled)
        # x: Skip connection (from encoder)

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class ResUNet1D(nn.Module):
    """
    Phase-Aware Stratified 1D Attention ResUNet with ASPP.
    """

    def __init__(self):
        super(ResUNet1D, self).__init__()

        self.in_channels = Config.IN_CHANNELS
        self.out_channels = Config.OUT_CHANNELS
        self.base_filters = Config.BASE_FILTERS

        filters = [
            self.base_filters * (2**i) for i in range(5)
        ]  # [64, 128, 256, 512, 1024]

        # --- Encoder ---
        self.input_block = nn.Sequential(
            nn.Conv1d(
                self.in_channels, filters[0], kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm1d(filters[0]),
            nn.ReLU(inplace=True),
        )

        self.enc1 = ResidualBlock1D(filters[0], filters[0])
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = ResidualBlock1D(filters[0], filters[1])
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = ResidualBlock1D(filters[1], filters[2])
        self.pool3 = nn.MaxPool1d(2)

        self.enc4 = ResidualBlock1D(filters[2], filters[3])
        self.pool4 = nn.MaxPool1d(2)

        # --- Bottleneck (ASPP) ---
        self.aspp = ASPP(filters[3], filters[4], dilations=Config.ASPP_DILATIONS)

        # --- Decoder ---
        # Up 4
        self.up4 = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.att4 = AttentionGate(F_g=filters[4], F_l=filters[3], F_int=filters[3] // 2)
        self.dec4 = ResidualBlock1D(filters[4] + filters[3], filters[3])

        # Up 3
        self.up3 = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.att3 = AttentionGate(F_g=filters[3], F_l=filters[2], F_int=filters[2] // 2)
        self.dec3 = ResidualBlock1D(filters[3] + filters[2], filters[2])

        # Up 2
        self.up2 = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.att2 = AttentionGate(F_g=filters[2], F_l=filters[1], F_int=filters[1] // 2)
        self.dec2 = ResidualBlock1D(filters[2] + filters[1], filters[1])

        # Up 1
        self.up1 = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.att1 = AttentionGate(F_g=filters[1], F_l=filters[0], F_int=filters[0] // 2)
        self.dec1 = ResidualBlock1D(filters[1] + filters[0], filters[0])

        # --- Heads ---
        self.final_head = nn.Conv1d(filters[0], self.out_channels, kernel_size=1)

        # Auxiliary heads for Deep Supervision (Decimated targets)
        self.aux_head2 = nn.Conv1d(filters[1], self.out_channels, kernel_size=1)
        self.aux_head3 = nn.Conv1d(filters[2], self.out_channels, kernel_size=1)
        self.aux_head4 = nn.Conv1d(filters[3], self.out_channels, kernel_size=1)

    def forward(self, x):
        # x: (Batch, Channels, Time)

        # Encoder
        x1 = self.input_block(x)
        x1 = self.enc1(x1)
        p1 = self.pool1(x1)

        x2 = self.enc2(p1)
        p2 = self.pool2(x2)

        x3 = self.enc3(p2)
        p3 = self.pool3(x3)

        x4 = self.enc4(p3)
        p4 = self.pool4(x4)

        # Bottleneck
        b = self.aspp(p4)

        # Decoder
        # D4
        d4 = self.up4(b)
        # Handle padding issues if input size was odd
        if d4.size(2) != x4.size(2):
            d4 = F.interpolate(d4, size=x4.size(2), mode="linear", align_corners=True)

        a4 = self.att4(g=d4, x=x4)
        d4 = torch.cat([d4, a4], dim=1)
        d4 = self.dec4(d4)
        out_aux4 = self.aux_head4(d4)

        # D3
        d3 = self.up3(d4)
        if d3.size(2) != x3.size(2):
            d3 = F.interpolate(d3, size=x3.size(2), mode="linear", align_corners=True)

        a3 = self.att3(g=d3, x=x3)
        d3 = torch.cat([d3, a3], dim=1)
        d3 = self.dec3(d3)
        out_aux3 = self.aux_head3(d3)

        # D2
        d2 = self.up2(d3)
        if d2.size(2) != x2.size(2):
            d2 = F.interpolate(d2, size=x2.size(2), mode="linear", align_corners=True)

        a2 = self.att2(g=d2, x=x2)
        d2 = torch.cat([d2, a2], dim=1)
        d2 = self.dec2(d2)
        out_aux2 = self.aux_head2(d2)

        # D1
        d1 = self.up1(d2)
        if d1.size(2) != x1.size(2):
            d1 = F.interpolate(d1, size=x1.size(2), mode="linear", align_corners=True)

        a1 = self.att1(g=d1, x=x1)
        d1 = torch.cat([d1, a1], dim=1)
        d1 = self.dec1(d1)

        # Final Output
        out_final = self.final_head(d1)

        return out_final, [out_aux2, out_aux3, out_aux4]
