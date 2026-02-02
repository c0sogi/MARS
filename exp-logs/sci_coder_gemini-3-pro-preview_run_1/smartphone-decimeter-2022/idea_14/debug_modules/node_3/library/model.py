import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResBlock1D(nn.Module):
    """
    1D Residual Block with optional downsampling.
    Structure: Conv1D -> BN -> ReLU -> Conv1D -> BN -> Residual Add -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        # Calculate padding to keep dimensions consistent (if stride=1)
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.
    Captures multi-scale context using parallel dilated convolutions.
    """

    def __init__(self, in_channels, out_channels, dilations):
        super().__init__()
        self.modules_list = nn.ModuleList()

        # Branch 1: 1x1 Conv
        self.modules_list.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Branches 2-N: Dilated Convs
        for dilation in dilations:
            # Padding to maintain sequence length
            padding = dilation
            self.modules_list.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        3,
                        padding=padding,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Branch N+1: Global Average Pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Final projection layer
        # Input channels = out_channels * (number of dilation branches + 1x1 branch + global pool branch)
        num_branches = len(dilations) + 2
        self.project = nn.Sequential(
            nn.Conv1d(out_channels * num_branches, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        # Convolutions
        for mod in self.modules_list:
            res.append(mod(x))

        # Global pooling
        gap = self.global_avg_pool(x)
        gap = F.interpolate(gap, size=x.size(2), mode="nearest")
        res.append(gap)

        # Concatenate and project
        res = torch.cat(res, dim=1)
        return self.project(res)


class ResUNet1D(nn.Module):
    """
    1D Residual U-Net with ASPP bottleneck and Deep Supervision.
    Predicts 2D residual vectors (North, East) from GNSS features.
    """

    def __init__(self):
        super().__init__()

        self.in_channels = Config.IN_CHANNELS
        self.out_channels = Config.OUT_CHANNELS
        base_c = Config.BASE_CHANNELS

        # --- Encoder ---
        # Stem: Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv1d(self.in_channels, base_c, 3, padding=1, bias=False),
            nn.BatchNorm1d(base_c),
            nn.ReLU(inplace=True),
        )

        # Encoder Stages (Downsampling)
        # Stage 1
        self.enc1 = ResBlock1D(base_c, base_c)
        self.down1 = nn.MaxPool1d(2)

        # Stage 2
        self.enc2 = ResBlock1D(base_c, base_c * 2)
        self.down2 = nn.MaxPool1d(2)

        # Stage 3
        self.enc3 = ResBlock1D(base_c * 2, base_c * 4)
        self.down3 = nn.MaxPool1d(2)

        # Stage 4
        self.enc4 = ResBlock1D(base_c * 4, base_c * 8)
        self.down4 = nn.MaxPool1d(2)

        # --- Bottleneck ---
        self.aspp = ASPP(base_c * 8, base_c * 8, Config.ASPP_DILATIONS)

        # --- Decoder ---
        # Stage 4 (Up)
        self.up4 = nn.ConvTranspose1d(base_c * 8, base_c * 4, 2, stride=2)
        # Input to dec4 is cat(up4_out, enc4_out) = 256 + 512 = 768 channels
        self.dec4 = ResBlock1D(base_c * 12, base_c * 4)

        # Stage 3 (Up)
        self.up3 = nn.ConvTranspose1d(base_c * 4, base_c * 2, 2, stride=2)
        # Input to dec3 is cat(up3_out, enc3_out) = 128 + 256 = 384 channels
        self.dec3 = ResBlock1D(base_c * 6, base_c * 2)

        # Stage 2 (Up)
        self.up2 = nn.ConvTranspose1d(base_c * 2, base_c, 2, stride=2)
        # Input to dec2 is cat(up2_out, enc2_out) = 64 + 128 = 192 channels
        self.dec2 = ResBlock1D(base_c * 3, base_c)

        # Stage 1 (Up)
        self.up1 = nn.ConvTranspose1d(base_c, base_c, 2, stride=2)
        self.dec1 = ResBlock1D(base_c * 2, base_c)  # Input: cat(up1, enc1)

        # --- Heads ---
        # Main Output Head (Full Resolution)
        self.head_main = nn.Conv1d(base_c, self.out_channels, 1)

        # Auxiliary Heads for Deep Supervision
        # Aux 1 connected to Decoder Stage 3 (Resolution / 4)
        self.head_aux1 = nn.Conv1d(base_c * 2, self.out_channels, 1)

        # Aux 2 connected to Decoder Stage 2 (Resolution / 2)
        self.head_aux2 = nn.Conv1d(base_c, self.out_channels, 1)

    def forward(self, x):
        # x shape: (Batch, In_Channels, Length)

        # --- Encoder Pass ---
        x0 = self.stem(x)  # Length: L

        x1 = self.enc1(x0)  # L
        p1 = self.down1(x1)  # L/2

        x2 = self.enc2(p1)  # L/2
        p2 = self.down2(x2)  # L/4

        x3 = self.enc3(p2)  # L/4
        p3 = self.down3(x3)  # L/8

        x4 = self.enc4(p3)  # L/8
        p4 = self.down4(x4)  # L/16

        # --- Bottleneck ---
        b = self.aspp(p4)  # L/16

        # --- Decoder Pass ---
        # Stage 4
        d4 = self.up4(b)  # L/8
        # Handle padding mismatches due to odd input lengths
        if d4.size(2) != x4.size(2):
            d4 = F.interpolate(d4, size=x4.size(2), mode="linear", align_corners=False)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dec4(d4)

        # Stage 3
        d3 = self.up3(d4)  # L/4
        if d3.size(2) != x3.size(2):
            d3 = F.interpolate(d3, size=x3.size(2), mode="linear", align_corners=False)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)

        # Stage 2
        d2 = self.up2(d3)  # L/2
        if d2.size(2) != x2.size(2):
            d2 = F.interpolate(d2, size=x2.size(2), mode="linear", align_corners=False)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        # Stage 1
        d1 = self.up1(d2)  # L
        if d1.size(2) != x1.size(2):
            d1 = F.interpolate(d1, size=x1.size(2), mode="linear", align_corners=False)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)

        # --- Output Heads ---
        out_main = self.head_main(d1)

        if self.training:
            # Compute auxiliary outputs
            out_aux1 = self.head_aux1(d3)
            out_aux2 = self.head_aux2(d2)

            # Upsample auxiliary outputs to match ground truth length (L)
            target_len = x.size(2)
            out_aux1 = F.interpolate(
                out_aux1, size=target_len, mode="linear", align_corners=False
            )
            out_aux2 = F.interpolate(
                out_aux2, size=target_len, mode="linear", align_corners=False
            )

            return out_main, out_aux1, out_aux2

        return out_main
