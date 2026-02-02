import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResBlock1D(nn.Module):
    """
    1D Residual Block with optional Dropout.
    Structure: Conv1d -> BN -> ReLU -> Dropout -> Conv1d -> BN -> Add -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        super(ResBlock1D, self).__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection to match dimensions
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, bias=False
            )

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = self.act(x)
        return x


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for 1D sequences.
    Captures multi-scale context using different dilation rates.
    """

    def __init__(self, in_channels, out_channels, dilations):
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

        # Branches 2..N: Dilated Convs
        for rate in dilations:
            # For kernel=3, padding=rate preserves length
            self.branches.append(
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

        # Branch N+1: Global Average Pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Final projection
        # Input to project is out_channels * (len(dilations) + 2)
        total_branches = len(dilations) + 2
        self.project = nn.Sequential(
            nn.Conv1d(out_channels * total_branches, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
        )

    def forward(self, x):
        res = []
        # Convolutions
        for branch in self.branches:
            res.append(branch(x))

        # Global Pooling
        # Expand global pool result to match input length
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[-1], mode="nearest")
        res.append(global_feat)

        # Concatenate and project
        res = torch.cat(res, dim=1)
        return self.project(res)


class AuxiliaryHead(nn.Module):
    """
    Lightweight head for Deep Supervision.
    """

    def __init__(self, in_channels, out_channels=2):
        super(AuxiliaryHead, self).__init__()
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, out_channels, 1),
        )

    def forward(self, x):
        return self.head(x)


class StratifiedResUNet1D(nn.Module):
    """
    1D Residual U-Net with Stratified Inputs and Decimated Deep Supervision.
    """

    def __init__(self):
        super(StratifiedResUNet1D, self).__init__()

        self.in_channels = Config.IN_CHANNELS
        self.base_filters = Config.NUM_FILTERS
        self.depth = Config.ENCODER_DEPTH

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        current_c = self.in_channels
        for i in range(self.depth):
            out_c = self.base_filters * (2**i)
            block = ResBlock1D(
                current_c,
                out_c,
                kernel_size=Config.KERNEL_SIZE,
                dropout=Config.DROPOUT_RATE,
            )
            self.encoder_blocks.append(block)
            self.downsamples.append(nn.MaxPool1d(2))
            current_c = out_c

        # --- Bottleneck (ASPP) ---
        # We double channels at bottleneck relative to last encoder
        bottleneck_in = current_c
        bottleneck_out = current_c * 2
        self.aspp = ASPP(bottleneck_in, bottleneck_out, Config.ASPP_DILATIONS)
        current_c = bottleneck_out

        # --- Decoder ---
        self.up_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.aux_heads = nn.ModuleDict()

        # Iterate backwards from depth-1 to 0
        for i in range(self.depth - 1, -1, -1):
            skip_c = self.base_filters * (2**i)

            # Upsample
            self.up_convs.append(
                nn.ConvTranspose1d(current_c, skip_c, kernel_size=2, stride=2)
            )

            # Decoder Block (Input is skip + upsampled, so 2 * skip_c)
            # We reduce back to skip_c
            self.decoder_blocks.append(
                ResBlock1D(
                    skip_c * 2,
                    skip_c,
                    kernel_size=Config.KERNEL_SIZE,
                    dropout=Config.DROPOUT_RATE,
                )
            )

            current_c = skip_c

            # Deep Supervision
            # The stride at this level relative to input is 2^i
            stride = 2**i
            if stride in Config.DEEP_SUPERVISION_STRIDES:
                self.aux_heads[str(stride)] = AuxiliaryHead(current_c, 2)

        # --- Final Head ---
        self.final_head = nn.Conv1d(current_c, 2, kernel_size=1)

    def forward(self, x):
        # x: (Batch, Channels, Length)

        # Encoder Pass
        skips = []
        for block, down in zip(self.encoder_blocks, self.downsamples):
            x = block(x)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.aspp(x)

        outputs = {}

        # Decoder Pass
        # We iterate through up_convs and decoder_blocks
        # We need to pop skips from the end
        for i, (up, dec) in enumerate(zip(self.up_convs, self.decoder_blocks)):
            skip = skips[-(i + 1)]

            x = up(x)

            # Handle padding/length mismatch due to pooling odd lengths
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(
                    x, size=skip.shape[-1], mode="linear", align_corners=False
                )

            x = torch.cat([x, skip], dim=1)
            x = dec(x)

            # Aux Head Check
            # Current stride level corresponds to 2^(depth - 1 - i)
            # i=0 (first decode step) -> depth-1
            # ...
            # i=depth-1 (last step) -> 0 (stride 1)
            current_stride_power = self.depth - 1 - i
            current_stride = 2**current_stride_power

            if str(current_stride) in self.aux_heads:
                outputs[f"aux_{current_stride}"] = self.aux_heads[str(current_stride)](
                    x
                )

        # Final Output (Stride 1)
        outputs["main"] = self.final_head(x)

        return outputs
