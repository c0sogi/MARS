import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D data.
    Recalibrates channel importance based on global context.
    """

    def __init__(self, channels, reduction=Config.SE_REDUCTION):
        super(SEBlock, self).__init__()
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
    1D Residual Block with Squeeze-and-Excitation.
    Structure: Conv -> BN -> ReLU -> Conv -> BN -> SE -> Add -> ReLU
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=Config.KERNEL_SIZE, stride=1
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

        self.se = SEBlock(out_channels)

        # Shortcut connection
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

        out = self.se(out)

        out += residual
        out = self.relu(out)
        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for 1D.
    Captures multi-scale temporal dependencies.
    """

    def __init__(self, in_channels, out_channels, rates=Config.ASPP_RATES):
        super(ASPP, self).__init__()

        self.branches = nn.ModuleList()

        # 1x1 Conv branch
        self.branches.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Dilated Conv branches
        for rate in rates:
            if rate == 1:
                continue  # Handled by 1x1 conv above usually, but config has 1.
            # If config has 1, it's essentially 3x3 with rate 1.
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Global Average Pooling branch
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion
        # Total channels = out_channels * (len(rates) + 1 (global))
        # Note: If rates includes 1, we have 1x1, 3x3r1, 3x3r6... + global
        num_branches = len(self.branches) + 1
        self.project = nn.Sequential(
            nn.Conv1d(out_channels * num_branches, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for branch in self.branches:
            res.append(branch(x))

        # Global branch needs upsampling
        global_feat = self.global_branch(x)
        global_feat = F.interpolate(
            global_feat, size=x.shape[2], mode="linear", align_corners=False
        )
        res.append(global_feat)

        res = torch.cat(res, dim=1)
        return self.project(res)


class SEResUNet1D(nn.Module):
    """
    Main Model: 1D U-Net with SE-ResBlock Encoder and ASPP Bottleneck.
    Supports Deep Supervision.
    """

    def __init__(self, in_channels=Config.INPUT_CHANNELS, out_channels=2):
        super(SEResUNet1D, self).__init__()

        # --- Encoder ---
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                Config.ENCODER_CHANNELS[0],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(Config.ENCODER_CHANNELS[0]),
            nn.ReLU(inplace=True),
        )

        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # Build Encoder Stages
        # Input to first block is from stem (channel[0])
        # Subsequent blocks take input from prev block
        dims = Config.ENCODER_CHANNELS

        # Block 0: 64 -> 64
        self.enc_blocks.append(ResBlock1D(dims[0], dims[0]))
        self.downsamples.append(nn.MaxPool1d(2))

        # Block 1: 64 -> 128
        self.enc_blocks.append(ResBlock1D(dims[0], dims[1]))
        self.downsamples.append(nn.MaxPool1d(2))

        # Block 2: 128 -> 256
        self.enc_blocks.append(ResBlock1D(dims[1], dims[2]))
        self.downsamples.append(nn.MaxPool1d(2))

        # Block 3: 256 -> 512
        self.enc_blocks.append(ResBlock1D(dims[2], dims[3]))
        # No pooling after last encoder block, goes to ASPP

        # --- Bottleneck ---
        self.aspp = ASPP(dims[3], dims[3])

        # --- Decoder ---
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        dec_dims = Config.DECODER_CHANNELS  # [256, 128, 64, 32]

        # Dec 0: Input 512 (ASPP) + Skip 512 (Enc3) -> Out 256
        self.up_convs.append(
            nn.ConvTranspose1d(dims[3], dec_dims[0], kernel_size=2, stride=2)
        )
        self.dec_blocks.append(
            ResBlock1D(dec_dims[0] + dims[3], dec_dims[0])
        )  # Concat dim

        # Dec 1: Input 256 + Skip 256 (Enc2) -> Out 128
        self.up_convs.append(
            nn.ConvTranspose1d(dec_dims[0], dec_dims[1], kernel_size=2, stride=2)
        )
        self.dec_blocks.append(ResBlock1D(dec_dims[1] + dims[2], dec_dims[1]))

        # Dec 2: Input 128 + Skip 128 (Enc1) -> Out 64
        self.up_convs.append(
            nn.ConvTranspose1d(dec_dims[1], dec_dims[2], kernel_size=2, stride=2)
        )
        self.dec_blocks.append(ResBlock1D(dec_dims[2] + dims[1], dec_dims[2]))

        # Dec 3: Input 64 + Skip 64 (Enc0) -> Out 32
        self.up_convs.append(
            nn.ConvTranspose1d(dec_dims[2], dec_dims[3], kernel_size=2, stride=2)
        )
        self.dec_blocks.append(ResBlock1D(dec_dims[3] + dims[0], dec_dims[3]))

        # --- Heads ---
        # Final Head (Full Resolution) from Dec 3
        self.final_head = nn.Conv1d(dec_dims[3], out_channels, 1)

        # Aux Head 1 from Dec 2 (1/2 Resolution)
        self.aux_head1 = nn.Conv1d(dec_dims[2], out_channels, 1)

        # Aux Head 2 from Dec 1 (1/4 Resolution)
        self.aux_head2 = nn.Conv1d(dec_dims[1], out_channels, 1)

    def forward(self, x):
        # x: (B, C, T)

        # --- Encoder ---
        x = self.stem(x)

        skips = []

        # Enc 0
        x0 = self.enc_blocks[0](x)
        skips.append(x0)  # Skip for Dec 3
        x = self.downsamples[0](x0)

        # Enc 1
        x1 = self.enc_blocks[1](x)
        skips.append(x1)  # Skip for Dec 2
        x = self.downsamples[1](x1)

        # Enc 2
        x2 = self.enc_blocks[2](x)
        skips.append(x2)  # Skip for Dec 1
        x = self.downsamples[2](x2)

        # Enc 3
        x3 = self.enc_blocks[3](x)
        skips.append(x3)  # Skip for Dec 0
        # No pool

        # --- Bottleneck ---
        x = self.aspp(x3)

        # --- Decoder ---
        # Skips order: [x0(64), x1(128), x2(256), x3(512)]

        # Dec 0 (Uses x3)
        x = self.up_convs[0](x)
        # Handle potential padding issues if T is not power of 2
        if x.size(2) != skips[3].size(2):
            x = F.interpolate(
                x, size=skips[3].size(2), mode="linear", align_corners=False
            )
        x = torch.cat([x, skips[3]], dim=1)
        x = self.dec_blocks[0](x)

        # Dec 1 (Uses x2) -> Aux Head 2
        x = self.up_convs[1](x)
        if x.size(2) != skips[2].size(2):
            x = F.interpolate(
                x, size=skips[2].size(2), mode="linear", align_corners=False
            )
        x = torch.cat([x, skips[2]], dim=1)
        x = self.dec_blocks[1](x)
        out_aux2 = self.aux_head2(x)

        # Dec 2 (Uses x1) -> Aux Head 1
        x = self.up_convs[2](x)
        if x.size(2) != skips[1].size(2):
            x = F.interpolate(
                x, size=skips[1].size(2), mode="linear", align_corners=False
            )
        x = torch.cat([x, skips[1]], dim=1)
        x = self.dec_blocks[2](x)
        out_aux1 = self.aux_head1(x)

        # Dec 3 (Uses x0) -> Final Head
        x = self.up_convs[3](x)
        if x.size(2) != skips[0].size(2):
            x = F.interpolate(
                x, size=skips[0].size(2), mode="linear", align_corners=False
            )
        x = torch.cat([x, skips[0]], dim=1)
        x = self.dec_blocks[3](x)
        out_final = self.final_head(x)

        if self.training:
            return out_final, out_aux1, out_aux2
        else:
            return out_final
