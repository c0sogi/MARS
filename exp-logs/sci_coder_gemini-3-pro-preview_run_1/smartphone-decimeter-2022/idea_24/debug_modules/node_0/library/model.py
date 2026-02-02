import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResBlock1D(nn.Module):
    """
    1D Residual Block with Batch Normalization and ReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        super(ResBlock1D, self).__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

        # Projection for skip connection if dimensions change
        self.project = None
        if in_channels != out_channels:
            self.project = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, bias=False
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.project is not None:
            residual = self.project(residual)

        out += residual
        out = self.act2(out)
        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for 1D signals.
    Captures multi-scale context at the bottleneck.
    """

    def __init__(self, in_channels, out_channels, dilations=[1, 2, 4, 8]):
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
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels, out_channels, 3, padding=d, dilation=d, bias=False
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

        # Final projection
        # Total channels = (len(dilations) + 1 + 1) * out_channels
        total_in = (len(dilations) + 2) * out_channels
        self.project = nn.Sequential(
            nn.Conv1d(total_in, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        res = []
        for branch in self.branches:
            res.append(branch(x))

        # Global branch
        global_feat = self.global_branch(x)
        # Expand global feature to match input length
        global_feat = F.interpolate(global_feat, size=x.shape[-1], mode="nearest")
        res.append(global_feat)

        res = torch.cat(res, dim=1)
        return self.project(res)


class DecoderBlock(nn.Module):
    """
    Decoder block that upsamples, concatenates skip connections from both streams,
    reduces channels, and refines features.
    """

    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.0):
        super(DecoderBlock, self).__init__()

        # Total input channels = Upsampled Input + Skip A + Skip B
        # skip_channels is the channels of ONE skip stream (assuming symmetric encoders)
        total_in_channels = in_channels + (2 * skip_channels)

        self.reduce = nn.Sequential(
            nn.Conv1d(total_in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.res_block = ResBlock1D(out_channels, out_channels, dropout=dropout)

    def forward(self, x, skip_a, skip_b):
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)

        # Handle potential size mismatch due to odd padding/cropping in encoder
        if x.size(-1) != skip_a.size(-1):
            x = F.interpolate(
                x, size=skip_a.size(-1), mode="linear", align_corners=False
            )

        # Concatenate
        x = torch.cat([x, skip_a, skip_b], dim=1)

        # Process
        x = self.reduce(x)
        x = self.res_block(x)
        return x


class DualResUNet(nn.Module):
    """
    Dual-Stream 1D Residual U-Net with Decimated Deep Supervision.
    """

    def __init__(self):
        super(DualResUNet, self).__init__()
        self.config = Config

        base_filters = self.config.BASE_FILTERS
        depth = self.config.DEPTH

        # --- Encoders ---
        # We construct two identical encoders for Stream A and Stream B
        # Initial Projections
        self.start_conv_a = nn.Sequential(
            nn.Conv1d(self.config.IN_CHANNELS_A, base_filters, 1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.start_conv_b = nn.Sequential(
            nn.Conv1d(self.config.IN_CHANNELS_B, base_filters, 1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
        )

        self.encoder_blocks_a = nn.ModuleList()
        self.encoder_blocks_b = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # Build Encoder Layers
        # Levels 0 to Depth-1
        for i in range(depth):
            in_ch = base_filters * (2**i)
            out_ch = base_filters * (
                2 ** (i)
            )  # Keep same channels within block before pooling

            # Note: We keep channels constant in the ResBlock at level i,
            # then double them in the next level's input (handled by next block logic or bottleneck)
            # Actually, standard UNet doubles channels after downsampling.

            # Let's define: Block i processes features of size L/(2^i) with Channels * 2^i

            self.encoder_blocks_a.append(
                ResBlock1D(in_ch, in_ch, dropout=self.config.DROPOUT)
            )
            self.encoder_blocks_b.append(
                ResBlock1D(in_ch, in_ch, dropout=self.config.DROPOUT)
            )

            # Downsample layer doubles channels for the next stage
            self.downsamples.append(
                nn.Sequential(
                    nn.MaxPool1d(2),
                    nn.Conv1d(in_ch, in_ch * 2, 1, bias=False),
                    nn.BatchNorm1d(in_ch * 2),
                    nn.ReLU(inplace=True),
                )
            )

        # --- Bottleneck ---
        # Input to bottleneck is output of downsample(Level depth-1)
        # Channels = base * 2^depth
        bottleneck_in = base_filters * (2**depth)
        # We have two streams concatenated -> 2 * bottleneck_in

        # ASPP reduces channels back to bottleneck_in
        self.aspp = ASPP(bottleneck_in * 2, bottleneck_in)

        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()
        self.aux_head = None
        self.aux_layer_idx = -1

        # Build Decoder Layers
        # We go from Level depth-1 down to 0
        for i in range(depth - 1, -1, -1):
            # Input to decoder block comes from previous upsample (or bottleneck)
            # In channels = base * 2^(i+1)
            # Skip channels = base * 2^i
            # Out channels = base * 2^i

            in_ch = base_filters * (2 ** (i + 1))
            skip_ch = base_filters * (2**i)
            out_ch = skip_ch

            self.decoder_blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch, dropout=self.config.DROPOUT)
            )

            # Check for Auxiliary Head placement
            # Resolution at this stage (after upsampling) is L / 2^i
            # We want resolution L / Decimation_Factor
            # 2^i == Decimation_Factor
            current_downsample_factor = 2**i
            if current_downsample_factor == self.config.DECIMATION_FACTOR:
                self.aux_head = nn.Conv1d(out_ch, self.config.NUM_CLASSES, 1)
                self.aux_layer_idx = (depth - 1) - i  # Index in the decoder_blocks list

        # --- Final Output ---
        self.final_conv = nn.Conv1d(base_filters, self.config.NUM_CLASSES, 1)

    def forward(self, x_a, x_b):
        # --- Encoders ---
        # Initial conv
        xa = self.start_conv_a(x_a)
        xb = self.start_conv_b(x_b)

        skips_a = []
        skips_b = []

        # Encoder Stages
        for block_a, block_b, down in zip(
            self.encoder_blocks_a, self.encoder_blocks_b, self.downsamples
        ):
            # Process blocks
            xa = block_a(xa)
            xb = block_b(xb)

            # Save skips
            skips_a.append(xa)
            skips_b.append(xb)

            # Downsample
            xa = down(xa)
            xb = down(xb)

        # --- Bottleneck ---
        # Concatenate streams
        x = torch.cat([xa, xb], dim=1)
        x = self.aspp(x)

        # --- Decoder ---
        aux_out = None

        # Iterate backwards through skips
        # Skips are stored [Level0, Level1, ..., Level_Depth-1]
        # Decoder blocks process from Depth-1 down to 0

        for i, decoder_block in enumerate(self.decoder_blocks):
            # Pop corresponding skips
            s_a = skips_a.pop()
            s_b = skips_b.pop()

            x = decoder_block(x, s_a, s_b)

            # Check for Aux Head
            if i == self.aux_layer_idx and self.aux_head is not None:
                aux_out = self.aux_head(x)

        # --- Final Output ---
        out = self.final_conv(x)

        if self.training and aux_out is not None:
            return {"output": out, "aux": aux_out}
        else:
            return out
