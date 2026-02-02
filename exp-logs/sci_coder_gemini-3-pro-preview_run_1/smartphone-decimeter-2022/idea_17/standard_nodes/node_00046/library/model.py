import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with optional downsampling.
    Structure: Conv1d -> BN -> ReLU -> Dropout -> Conv1d -> BN -> Add -> ReLU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        dropout=0.0,
    ):
        super().__init__()
        # Calculate padding to maintain size (for stride=1) or proper downsampling
        padding = (kernel_size - 1) * dilation // 2

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
        self.dropout = nn.Dropout(dropout)

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
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for 1D data.
    Captures multi-scale context using different dilation rates.
    """

    def __init__(self, in_channels, out_channels, rates):
        super().__init__()
        modules = []

        # 1x1 Conv branch
        modules.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Atrous Conv branches
        for rate in rates:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Global Average Pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.convs = nn.ModuleList(modules)

        # Project back to desired output channels
        # Input channels to projection = (number of branches + global pool) * out_channels
        self.project = nn.Sequential(
            nn.Conv1d((len(rates) + 2) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        # Global pooling branch
        gap = self.global_avg_pool(x)
        # Interpolate back to input length
        gap = F.interpolate(gap, size=x.size(2), mode="nearest")
        res.append(gap)

        res = torch.cat(res, dim=1)
        return self.project(res)


class UNetStage1(nn.Module):
    """
    Stage 1: Deep Residual U-Net with ASPP and Auxiliary Heads.
    Performs coarse correction of global drift and low-frequency errors.
    """

    def __init__(self):
        super().__init__()

        # Config parameters
        in_dim = Config.INPUT_DIM
        enc_filters = Config.STAGE1_ENCODER_FILTERS  # e.g., [32, 64, 128, 256]
        dec_filters = Config.STAGE1_DECODER_FILTERS  # e.g., [256, 128, 64, 32]
        aspp_rates = Config.ASPP_RATES
        k_size = Config.KERNEL_SIZE
        dropout = Config.DROPOUT
        out_dim = Config.OUTPUT_DIM

        # --- Encoder ---
        # Initial projection
        self.enc0 = nn.Sequential(
            nn.Conv1d(in_dim, enc_filters[0], k_size, padding=k_size // 2, bias=False),
            nn.BatchNorm1d(enc_filters[0]),
            nn.ReLU(inplace=True),
        )

        # Downsampling blocks
        self.enc1 = ResidualBlock1D(
            enc_filters[0], enc_filters[1], k_size, stride=2, dropout=dropout
        )
        self.enc2 = ResidualBlock1D(
            enc_filters[1], enc_filters[2], k_size, stride=2, dropout=dropout
        )
        self.enc3 = ResidualBlock1D(
            enc_filters[2], enc_filters[3], k_size, stride=2, dropout=dropout
        )

        # --- Bottleneck ---
        self.aspp = ASPP(enc_filters[3], enc_filters[3], aspp_rates)

        # --- Decoder ---
        # Up 1 (from Bottleneck L/8 -> L/4)
        self.up1 = nn.ConvTranspose1d(
            enc_filters[3], dec_filters[1], kernel_size=2, stride=2
        )
        self.dec1 = ResidualBlock1D(
            dec_filters[1] + enc_filters[2], dec_filters[1], k_size, dropout=dropout
        )
        self.aux1 = nn.Conv1d(dec_filters[1], out_dim, 1)  # Aux head at 1/4 resolution

        # Up 2 (from Dec1 L/4 -> L/2)
        self.up2 = nn.ConvTranspose1d(
            dec_filters[1], dec_filters[2], kernel_size=2, stride=2
        )
        self.dec2 = ResidualBlock1D(
            dec_filters[2] + enc_filters[1], dec_filters[2], k_size, dropout=dropout
        )
        self.aux2 = nn.Conv1d(dec_filters[2], out_dim, 1)  # Aux head at 1/2 resolution

        # Up 3 (from Dec2 L/2 -> L)
        self.up3 = nn.ConvTranspose1d(
            dec_filters[2], dec_filters[3], kernel_size=2, stride=2
        )
        self.dec3 = ResidualBlock1D(
            dec_filters[3] + enc_filters[0], dec_filters[3], k_size, dropout=dropout
        )

        # Final Head
        self.final = nn.Conv1d(dec_filters[3], out_dim, 1)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)  # L
        e1 = self.enc1(e0)  # L/2
        e2 = self.enc2(e1)  # L/4
        e3 = self.enc3(e2)  # L/8

        # Bottleneck
        b = self.aspp(e3)  # L/8

        # Decoder
        # Block 1
        d1 = self.up1(b)  # L/4
        if d1.size(2) != e2.size(2):
            d1 = F.interpolate(d1, size=e2.size(2), mode="linear", align_corners=False)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        aux1 = self.aux1(d1)

        # Block 2
        d2 = self.up2(d1)  # L/2
        if d2.size(2) != e1.size(2):
            d2 = F.interpolate(d2, size=e1.size(2), mode="linear", align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        aux2 = self.aux2(d2)

        # Block 3
        d3 = self.up3(d2)  # L
        if d3.size(2) != e0.size(2):
            d3 = F.interpolate(d3, size=e0.size(2), mode="linear", align_corners=False)
        d3 = torch.cat([d3, e0], dim=1)
        d3 = self.dec3(d3)

        out = self.final(d3)

        return out, [aux1, aux2]


class UNetStage2(nn.Module):
    """
    Stage 2: Shallow Refinement U-Net.
    Takes original features + Stage 1 output to predict residual errors.
    """

    def __init__(self):
        super().__init__()

        # Input is Original Features + Stage 1 Output
        in_dim = Config.INPUT_DIM + Config.OUTPUT_DIM
        enc_filters = Config.STAGE2_ENCODER_FILTERS  # e.g., [32, 64]
        dec_filters = Config.STAGE2_DECODER_FILTERS  # e.g., [64, 32]
        k_size = Config.KERNEL_SIZE
        dropout = Config.DROPOUT
        out_dim = Config.OUTPUT_DIM

        # Encoder
        self.enc0 = nn.Sequential(
            nn.Conv1d(in_dim, enc_filters[0], k_size, padding=k_size // 2, bias=False),
            nn.BatchNorm1d(enc_filters[0]),
            nn.ReLU(inplace=True),
        )

        self.enc1 = ResidualBlock1D(
            enc_filters[0], enc_filters[1], k_size, stride=2, dropout=dropout
        )

        # Bottleneck
        self.bottleneck = ResidualBlock1D(
            enc_filters[1], enc_filters[1], k_size, stride=1, dropout=dropout
        )

        # Decoder
        self.up1 = nn.ConvTranspose1d(
            enc_filters[1], dec_filters[1], kernel_size=2, stride=2
        )
        self.dec1 = ResidualBlock1D(
            dec_filters[1] + enc_filters[0], dec_filters[1], k_size, dropout=dropout
        )

        self.final = nn.Conv1d(dec_filters[1], out_dim, 1)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)  # L
        e1 = self.enc1(e0)  # L/2

        # Bottleneck
        b = self.bottleneck(e1)  # L/2

        # Decoder
        d1 = self.up1(b)  # L
        if d1.size(2) != e0.size(2):
            d1 = F.interpolate(d1, size=e0.size(2), mode="linear", align_corners=False)
        d1 = torch.cat([d1, e0], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)

        return out


class CascadedResUNet(nn.Module):
    """
    Cascaded Architecture combining Stage 1 (Coarse) and Stage 2 (Fine).
    """

    def __init__(self):
        super().__init__()
        self.stage1 = UNetStage1()
        self.stage2 = UNetStage2()

    def forward(self, x):
        # x shape: (B, C, L)

        # --- Stage 1 ---
        s1_out, aux_outputs = self.stage1(x)

        # --- Stage 2 ---
        # Concatenate original input and Stage 1 prediction
        s2_in = torch.cat([x, s1_out], dim=1)
        s2_residual = self.stage2(s2_in)

        # --- Final Output ---
        # Add residual correction to coarse prediction
        final_out = s1_out + s2_residual

        return final_out, aux_outputs
