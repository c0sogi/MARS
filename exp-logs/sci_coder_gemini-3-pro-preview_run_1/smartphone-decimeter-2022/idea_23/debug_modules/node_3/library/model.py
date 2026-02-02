import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D signals.
    Recalibrates channel importance based on global temporal context.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: Fully Connected Layers
        y = self.fc(y).view(b, c, 1)
        # Scale
        return x * y


class ResBlock1D(nn.Module):
    """
    Residual Block with Squeeze-and-Excitation.
    Structure: Conv1 -> BN -> ReLU -> Conv2 -> BN -> SE -> Add -> ReLU
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, reduction=16
    ):
        super(ResBlock1D, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
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
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock1D(out_channels, reduction)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for 1D.
    Captures multi-scale temporal context using dilated convolutions.
    """

    def __init__(self, in_channels, out_channels, dilations):
        super(ASPP, self).__init__()
        self.modules_list = nn.ModuleList()

        # 1x1 Conv Branch
        self.modules_list.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Dilated 3x3 Conv Branches
        for dilation in dilations:
            self.modules_list.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Global Pooling Branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Projection Layer
        # Input channels = out_channels * (1 (1x1) + len(dilations) + 1 (global pool))
        total_branches = 1 + len(dilations) + 1

        self.project = nn.Sequential(
            nn.Conv1d(out_channels * total_branches, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for m in self.modules_list:
            res.append(m(x))

        # Global pool branch
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.size(2), mode="nearest")
        res.append(gp)

        res = torch.cat(res, dim=1)
        return self.project(res)


class SEResUNet1D(nn.Module):
    """
    1D U-Net with Squeeze-and-Excitation Residual Blocks and ASPP Bottleneck.
    Outputs coordinate residuals at multiple resolutions for deep supervision.
    """

    def __init__(self, config: Config):
        super(SEResUNet1D, self).__init__()
        self.config = config

        # --- Encoder ---
        # Initial Convolution
        self.enc_start = nn.Sequential(
            nn.Conv1d(
                config.INPUT_CHANNELS,
                config.ENCODER_CHANNELS[0],
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(config.ENCODER_CHANNELS[0]),
            nn.ReLU(inplace=True),
        )

        # Encoder ResBlocks with Downsampling
        self.enc_blocks = nn.ModuleList()
        in_ch = config.ENCODER_CHANNELS[0]

        # Create blocks: 32->64, 64->128, 128->256
        for out_ch in config.ENCODER_CHANNELS[1:]:
            self.enc_blocks.append(
                ResBlock1D(in_ch, out_ch, stride=2, reduction=config.SE_REDUCTION)
            )
            in_ch = out_ch

        # --- Bottleneck ---
        self.aspp = ASPP(
            config.ENCODER_CHANNELS[-1],
            config.ENCODER_CHANNELS[-1],
            config.ASPP_DILATIONS,
        )

        # --- Decoder ---
        self.dec_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # We perform 3 upsampling steps to restore resolution from L/8 to L
        # Step 1: L/8 -> L/4. Input: Bottleneck(256) + Skip(128) -> Out(128)
        self.upsamples.append(
            nn.ConvTranspose1d(
                config.ENCODER_CHANNELS[-1], config.ENCODER_CHANNELS[-2], 2, stride=2
            )
        )
        self.dec_blocks.append(
            ResBlock1D(
                config.ENCODER_CHANNELS[-2] * 2,
                config.ENCODER_CHANNELS[-2],
                reduction=config.SE_REDUCTION,
            )
        )

        # Step 2: L/4 -> L/2. Input: Dec1(128) + Skip(64) -> Out(64)
        self.upsamples.append(
            nn.ConvTranspose1d(
                config.ENCODER_CHANNELS[-2], config.ENCODER_CHANNELS[-3], 2, stride=2
            )
        )
        self.dec_blocks.append(
            ResBlock1D(
                config.ENCODER_CHANNELS[-3] * 2,
                config.ENCODER_CHANNELS[-3],
                reduction=config.SE_REDUCTION,
            )
        )

        # Step 3: L/2 -> L. Input: Dec2(64) + Skip(32) -> Out(32)
        self.upsamples.append(
            nn.ConvTranspose1d(
                config.ENCODER_CHANNELS[-3], config.ENCODER_CHANNELS[-4], 2, stride=2
            )
        )
        self.dec_blocks.append(
            ResBlock1D(
                config.ENCODER_CHANNELS[-4] * 2,
                config.ENCODER_CHANNELS[-4],
                reduction=config.SE_REDUCTION,
            )
        )

        # --- Prediction Heads ---
        # Deep Supervision Heads
        self.head0 = nn.Conv1d(
            config.ENCODER_CHANNELS[-1], config.OUTPUT_CHANNELS, 1
        )  # From Bottleneck (L/8)
        self.head1 = nn.Conv1d(
            config.ENCODER_CHANNELS[-2], config.OUTPUT_CHANNELS, 1
        )  # From Dec1 (L/4)
        self.head2 = nn.Conv1d(
            config.ENCODER_CHANNELS[-3], config.OUTPUT_CHANNELS, 1
        )  # From Dec2 (L/2)
        self.head3 = nn.Conv1d(
            config.ENCODER_CHANNELS[-4], config.OUTPUT_CHANNELS, 1
        )  # From Dec3 (L) - Final

    def forward(self, x):
        # Encoder Path
        x0 = self.enc_start(x)  # (B, 32, L)
        x1 = self.enc_blocks[0](x0)  # (B, 64, L/2)
        x2 = self.enc_blocks[1](x1)  # (B, 128, L/4)
        x3 = self.enc_blocks[2](x2)  # (B, 256, L/8)

        # Bottleneck
        b = self.aspp(x3)  # (B, 256, L/8)

        # Decoder Path
        # D1: Up(Bottleneck) + x2
        u1 = self.upsamples[0](b)
        if u1.size(2) != x2.size(2):
            u1 = F.interpolate(u1, size=x2.size(2), mode="linear", align_corners=False)
        d1 = self.dec_blocks[0](torch.cat([u1, x2], dim=1))  # (B, 128, L/4)

        # D2: Up(D1) + x1
        u2 = self.upsamples[1](d1)
        if u2.size(2) != x1.size(2):
            u2 = F.interpolate(u2, size=x1.size(2), mode="linear", align_corners=False)
        d2 = self.dec_blocks[1](torch.cat([u2, x1], dim=1))  # (B, 64, L/2)

        # D3: Up(D2) + x0
        u3 = self.upsamples[2](d2)
        if u3.size(2) != x0.size(2):
            u3 = F.interpolate(u3, size=x0.size(2), mode="linear", align_corners=False)
        d3 = self.dec_blocks[2](torch.cat([u3, x0], dim=1))  # (B, 32, L)

        # Prediction Heads
        out0 = self.head0(b)
        out1 = self.head1(d1)
        out2 = self.head2(d2)
        out3 = self.head3(d3)

        return [out0, out1, out2, out3]
