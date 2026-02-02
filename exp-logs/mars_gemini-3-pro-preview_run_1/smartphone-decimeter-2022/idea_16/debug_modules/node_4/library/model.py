import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with two convolution layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock1D, self).__init__()

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

        # Shortcut connection to match dimensions if necessary
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
    Atrous Spatial Pyramid Pooling (ASPP) for 1D signals.
    Captures multi-scale context using varying dilation rates.
    """

    def __init__(self, in_channels, out_channels, dilations):
        super(ASPP, self).__init__()

        self.modules_list = nn.ModuleList()

        # 1x1 Convolution
        self.modules_list.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Dilated Convolutions
        for dilation in dilations:
            if dilation == 1:
                continue  # Already handled by 1x1 conv conceptually, but config might include 1.
                # If config has 1, it's a 3x3 with dilation 1.
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

        # Global Average Pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

        # Projection
        # Input channels = out_channels * (len(dilations) + 1 for global pool + 1 for 1x1 if not in dilations)
        # Adjusting logic: Config.ASPP_DILATIONS usually includes 1 or not.
        # Let's strictly follow the list + global pool.
        # If 1 is in dilations, we treat it as 3x3 dilation 1.
        # The first 1x1 conv is standard ASPP branch.

        # Cite debug_lesson_15: Account for Skip Connection Concatenation (here ASPP branches)
        # We must count only the actual branches added. Dilation 1 is skipped in the loop.
        actual_dilations = [d for d in dilations if d != 1]
        num_branches = 1 + len(actual_dilations) + 1  # 1x1 + dilated + global pool

        self.project = nn.Sequential(
            nn.Conv1d(out_channels * num_branches, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []

        # Branches
        for conv in self.modules_list:
            res.append(conv(x))

        # Global Pooling
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=x.size(2), mode="nearest")
        res.append(global_feat)

        res = torch.cat(res, dim=1)
        return self.project(res)


class ResUNet1D(nn.Module):
    """
    1D Residual U-Net with ASPP and Scaled Deep Supervision.
    """

    def __init__(self):
        super(ResUNet1D, self).__init__()

        self.in_channels = Config.IN_CHANNELS
        self.out_channels = Config.OUT_CHANNELS
        self.start_filters = Config.START_FILTERS
        self.depth = Config.ENCODER_DEPTH

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # Track channel progression for decoder symmetry
        # Stem output
        self.enc_channels = [self.start_filters]

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv1d(self.in_channels, self.start_filters, 3, padding=1, bias=False),
            nn.BatchNorm1d(self.start_filters),
            nn.ReLU(inplace=True),
        )

        in_ch = self.start_filters

        for i in range(self.depth):
            # i=0: 32->32. i=1: 32->64. i=2: 64->128...
            if i == 0:
                out_ch = self.start_filters
            else:
                out_ch = self.start_filters * (2**i)

            self.encoder_blocks.append(ResidualBlock1D(in_ch, out_ch))
            self.downsamples.append(nn.MaxPool1d(2))
            self.enc_channels.append(out_ch)
            in_ch = out_ch

        # --- Bottleneck ---
        # ASPP at the bottom
        self.aspp = ASPP(in_ch, in_ch, Config.ASPP_DILATIONS)

        # --- Decoder ---
        self.up_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # i goes from depth-1 down to 0
        for i in range(self.depth - 1, -1, -1):
            # Input channels:
            # If i == depth-1, input is bottleneck (same as last encoder output)
            # Else, input is output of previous decoder block (level i+1)
            if i == self.depth - 1:
                prev_ch = self.enc_channels[-1]
            else:
                # Output of decoder block at level i+1 matches skip_ch of level i+1
                prev_ch = self.enc_channels[i + 2]

            skip_ch = self.enc_channels[i + 1]
            out_ch = skip_ch

            self.up_convs.append(nn.ConvTranspose1d(prev_ch, skip_ch, 2, stride=2))

            # Residual block after concatenation
            # Cite debug_lesson_15: Account for Skip Connection Concatenation
            self.decoder_blocks.append(ResidualBlock1D(skip_ch * 2, out_ch))

        # --- Heads ---
        self.heads = nn.ModuleDict()

        # Head indices map to decoder block indices in the loop (0 to depth-1)
        # j=0 is bottom (lowest res), j=depth-1 is top (full res)
        self.head_indices = {0: self.depth - 1, 1: self.depth - 2, 2: self.depth - 3}

        for name, list_idx in self.head_indices.items():
            # Determine resolution level 'i' from decoder index 'list_idx'
            # list_idx corresponds to the loop iteration.
            # The loop variable 'i' was (depth-1 ... 0).
            # The list_idx-th block corresponds to loop variable i = depth - 1 - list_idx.

            i = self.depth - 1 - list_idx
            ch = self.enc_channels[i + 1]

            self.heads[str(name)] = nn.Conv1d(ch, self.out_channels, 1)

    def forward(self, x):
        # x: (B, C, L)

        # Encoder
        skips = []
        x = self.stem(x)
        skips.append(x)

        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            skips.append(x)
            x = self.downsamples[i](x)

        # Bottleneck
        x = self.aspp(x)

        # Decoder
        outputs = {}

        for i, (up, block) in enumerate(zip(self.up_convs, self.decoder_blocks)):
            skip = skips.pop()

            x = up(x)

            # Resize x to match skip if necessary (handling odd lengths)
            if x.size(2) != skip.size(2):
                x = F.interpolate(
                    x, size=skip.size(2), mode="linear", align_corners=False
                )

            x = torch.cat([x, skip], dim=1)
            x = block(x)

            # Check if this layer has a head attached
            if i in self.head_indices.values():
                # Find which head key corresponds to this index
                for key, val in self.head_indices.items():
                    if val == i:
                        outputs[key] = self.heads[str(key)](x)

        # Return list of outputs [Head 0, Head 1, Head 2]
        sorted_outputs = [outputs[k] for k in sorted(outputs.keys())]

        return sorted_outputs
