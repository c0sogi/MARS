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
            nn.BatchNorm1d(out_channels),
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

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv1d(self.in_channels, self.start_filters, 3, padding=1, bias=False),
            nn.BatchNorm1d(self.start_filters),
            nn.ReLU(inplace=True),
        )

        in_ch = self.start_filters

        for i in range(self.depth):
            out_ch = in_ch * 2 if i < 4 else in_ch  # Cap growth or standard doubling
            # Standard doubling strategy: 32 -> 64 -> 128 -> 256 -> 512
            out_ch = self.start_filters * (2**i)

            self.encoder_blocks.append(ResidualBlock1D(in_ch, out_ch))
            self.downsamples.append(nn.MaxPool1d(2))
            in_ch = out_ch

        # --- Bottleneck ---
        # ASPP at the bottom
        self.aspp = ASPP(in_ch, in_ch, Config.ASPP_DILATIONS)

        # --- Decoder ---
        self.up_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # We go from depth-1 down to 0
        # Encoder outputs to skip: indices 0 to depth-1
        # Bottleneck output is input to first upsample

        for i in range(self.depth - 1, -1, -1):
            # Input channels from previous layer (or bottleneck)
            prev_ch = (
                self.start_filters * (2**i)
                if i < self.depth - 1
                else self.start_filters * (2 ** (self.depth - 1))
            )

            # Skip connection channels
            skip_ch = self.start_filters * (2**i)

            # Output channels for this block
            out_ch = skip_ch  # We decode back to same dim as encoder level

            # Upsampling layer
            self.up_convs.append(nn.ConvTranspose1d(prev_ch, skip_ch, 2, stride=2))

            # Residual block after concatenation
            # Input to block is skip_ch + skip_ch
            self.decoder_blocks.append(ResidualBlock1D(skip_ch * 2, out_ch))

        # --- Heads ---
        # Deep supervision heads at specific resolutions
        # Head 0: Final (L) -> Index corresponds to last decoder block
        # Head 1: L/2 -> Second to last
        # Head 2: L/4 -> Third to last

        self.heads = nn.ModuleDict()

        # Decoder blocks list is ordered from bottom (lowest res) to top (highest res)
        # Index 0: Bottom-most decoder block (Resolution L / 2^(depth-1))
        # Index depth-1: Top-most decoder block (Resolution L)

        # We need heads at:
        # Final (L): Index depth-1
        # L/2: Index depth-2
        # L/4: Index depth-3

        self.head_indices = {0: self.depth - 1, 1: self.depth - 2, 2: self.depth - 3}

        for name, idx in self.head_indices.items():
            # Determine input channels for the head
            # The output of decoder_blocks[idx] has channels = start_filters * 2^(depth - 1 - idx)
            # Let's calculate properly based on loop above
            # Loop variable i went depth-1 down to 0.
            # decoder_blocks[0] corresponds to i = depth-1 (Resolution L/16 -> L/8)
            # decoder_blocks[last] corresponds to i = 0 (Resolution L/2 -> L)

            # Actually, let's map i to decoder_block index
            # j = 0 corresponds to i = depth-1
            # j corresponds to i = depth - 1 - j

            # We want resolution L (i=0). This is j = depth - 1.
            # We want resolution L/2 (i=1). This is j = depth - 2.
            # We want resolution L/4 (i=2). This is j = depth - 3.

            # Channels at i: start_filters * 2^i
            ch = self.start_filters * (2 ** (self.depth - 1 - idx))  # Wait, logic check

            # i=0 (Final), idx=depth-1. Channels = start * 2^0 = 32. Correct.
            # i=1 (L/2), idx=depth-2. Channels = start * 2^1 = 64. Correct.

            self.heads[str(name)] = nn.Conv1d(ch, self.out_channels, 1)

    def forward(self, x):
        # x: (B, C, L)

        # Encoder
        skips = []
        x = self.stem(x)
        skips.append(x)  # Resolution L, idx 0

        for i in range(self.depth):
            # Apply ResBlock then Pool
            # Note: In standard UNet, we crop/copy before pooling.
            # Here, encoder_blocks[i] processes the features at current resolution
            x = self.encoder_blocks[i](x)
            if i < self.depth - 1:
                skips.append(x)  # Store for skip connection
                x = self.downsamples[i](x)
            else:
                # Last block, just pool for bottleneck
                x = self.downsamples[i](x)

        # Bottleneck
        x = self.aspp(x)

        # Decoder
        outputs = {}

        for i, (up, block) in enumerate(zip(self.up_convs, self.decoder_blocks)):
            # Upsample
            x = up(x)

            # Get skip connection
            # skips list has: [Stem(L), Enc0(L), Enc1(L/2), Enc2(L/4), Enc3(L/8)]
            # We are decoding.
            # i=0: Bottom decoder. Needs Enc3 (L/8).
            # skips index = depth - 2 - i?
            # Let's trace:
            # Depth=5. Skips indices: 0(L), 1(L), 2(L/2), 3(L/4), 4(L/8).
            # Encoder structure in loop:
            # i=0: Enc0(L) -> Pool -> L/2
            # i=1: Enc1(L/2) -> Pool -> L/4
            # i=2: Enc2(L/4) -> Pool -> L/8
            # i=3: Enc3(L/8) -> Pool -> L/16
            # i=4: Enc4(L/16) -> Pool -> L/32 (Bottleneck)

            # Correct Skips collection:
            # We want the output of the block BEFORE pooling.
            # My loop above appends AFTER block.
            # skips[0] = Stem (L) - usually not used as skip in ResUNet, usually Enc0 output is used.
            # Let's adjust Encoder loop to be cleaner.
            pass

        # Re-running logic inside forward for clarity
        skips = []
        x = self.stem(x)  # (B, 32, L)

        # i=0: ResBlock(32->32) -> x_enc0 (L). Skip. Pool -> (L/2)
        # i=1: ResBlock(32->64) -> x_enc1 (L/2). Skip. Pool -> (L/4)
        # ...

        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            if i < self.depth - 1:
                skips.append(x)
                x = self.downsamples[i](x)
            else:
                # Last encoder block (L/16).
                # We need this for the first decoder step skip connection?
                # Usually U-Net bottleneck connects L/32 back to L/16.
                # The skip comes from the encoder level L/16.
                skips.append(x)
                x = self.downsamples[i](x)  # To bottleneck (L/32)

        x = self.aspp(x)

        # Decoder
        # Skips: [L, L/2, L/4, L/8, L/16]
        # We iterate backwards through skips

        for i, (up, block) in enumerate(zip(self.up_convs, self.decoder_blocks)):
            skip = skips.pop()

            # Handle size mismatch due to odd padding/pooling
            if x.size(2) != skip.size(2) // 2:
                # This might happen with TransposeConv.
                # Usually we pad x to match skip after upsampling.
                pass

            x = up(x)

            # Resize x to match skip if necessary (handling odd lengths)
            if x.size(2) != skip.size(2):
                x = F.interpolate(
                    x, size=skip.size(2), mode="linear", align_corners=False
                )

            x = torch.cat([x, skip], dim=1)
            x = block(x)

            # Check if this layer has a head attached
            # i=0 -> processing L/16 -> L/8. Output is L/8.
            # We want heads at L, L/2, L/4.
            # L is at end.

            # Mapping from loop index 'i' to resolution:
            # i=0: Output L/8
            # i=1: Output L/4  -> Head 2
            # i=2: Output L/2  -> Head 1
            # i=3: Output L    -> Head 0

            # Wait, depth=5.
            # Encoder: L -> L/2 -> L/4 -> L/8 -> L/16 -> (Pool) -> L/32
            # Skips: L, L/2, L/4, L/8, L/16
            # Dec i=0: Up(L/32)->L/16. Cat(L/16). Out L/16.
            # Dec i=1: Up(L/16)->L/8. Cat(L/8). Out L/8.
            # Dec i=2: Up(L/8)->L/4. Cat(L/4). Out L/4. (Head 2)
            # Dec i=3: Up(L/4)->L/2. Cat(L/2). Out L/2. (Head 1)
            # Dec i=4: Up(L/2)->L. Cat(L). Out L. (Head 0)

            # My indices in __init__ were:
            # Head 0: idx = depth-1 = 4
            # Head 1: idx = depth-2 = 3
            # Head 2: idx = depth-3 = 2

            if i in self.head_indices.values():
                # Find which head key corresponds to this index
                for key, val in self.head_indices.items():
                    if val == i:
                        outputs[key] = self.heads[str(key)](x)

        # Return list of outputs [Head 0, Head 1, Head 2]
        # Sort by key to ensure order
        sorted_outputs = [outputs[k] for k in sorted(outputs.keys())]

        return sorted_outputs
