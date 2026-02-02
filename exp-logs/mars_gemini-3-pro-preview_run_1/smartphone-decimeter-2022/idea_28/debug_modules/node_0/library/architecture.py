import torch
import torch.nn as nn
import torch.nn.functional as F
import library.config as config


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D signals.
    Recalibrates channel importance based on global temporal context.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        # Squeeze: Global Average Pooling over time
        y = self.avg_pool(x)
        # Excitation: Learn channel weights
        y = self.fc(y)
        # Scale
        return x * y


class ResBlock1D(nn.Module):
    """
    1D Residual Block with Batch Normalization and SE Attention.
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
        # Calculate padding to maintain sequence length (if stride=1)
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

        self.se = SEBlock1D(out_channels)
        self.dropout = nn.Dropout(dropout)

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
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        out += residual
        out = self.relu(out)
        return out


class ASPP1D(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for 1D.
    Captures context at multiple temporal scales.
    """

    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12, 18]):
        super().__init__()
        self.modules_list = nn.ModuleList()

        # 1x1 Conv
        self.modules_list.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Dilated Convs
        for dilation in dilations:
            # Padding = dilation for kernel_size=3 to maintain length
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

        # Global Pooling (Image Level Features)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Final Projection
        self.project = nn.Sequential(
            nn.Conv1d(out_channels * (len(dilations) + 2), out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        res = []
        for m in self.modules_list:
            res.append(m(x))

        # Global pool needs to be upsampled to input size
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.size(2), mode="linear", align_corners=False)
        res.append(gp)

        res = torch.cat(res, dim=1)
        return self.project(res)


class SEResUNet1D(nn.Module):
    """
    1D U-Net with SE-Residual Blocks and ASPP Bottleneck.
    Supports Decimated Deep Supervision.
    """

    def __init__(self):
        super().__init__()

        self.in_channels = config.INPUT_CHANNELS
        self.out_channels = config.OUTPUT_CHANNELS
        self.hidden_dim = config.HIDDEN_DIM
        self.depth = config.ENCODER_DEPTH
        self.aux_scales = config.AUXILIARY_SCALES

        # --- Encoder ---
        # Initial Stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.in_channels, self.hidden_dim, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        curr_dim = self.hidden_dim
        for i in range(self.depth):
            # Feature extraction at current resolution
            self.encoder_blocks.append(
                ResBlock1D(curr_dim, curr_dim, dropout=config.DROPOUT_RATE)
            )
            # Downsample for next level
            self.downsamples.append(
                nn.Conv1d(curr_dim, curr_dim * 2, kernel_size=3, stride=2, padding=1)
            )
            curr_dim *= 2

        # --- Bottleneck ---
        self.aspp = ASPP1D(curr_dim, curr_dim)

        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.aux_heads = nn.ModuleList()
        self.aux_indices = []  # Store indices of decoder steps that have aux heads

        # Decoder processes from bottom (bottleneck) up
        for i in range(self.depth):
            # Calculate the scale factor at this decoder level
            # i=0 is the deepest level (e.g. L/8), i=depth-1 is the final level (L)
            scale_factor = 2 ** (self.depth - 1 - i)

            # Upsample: curr_dim -> curr_dim // 2
            self.upsamples.append(
                nn.ConvTranspose1d(
                    curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1
                )
            )

            # Block takes concatenated input (curr_dim//2 + curr_dim//2) -> curr_dim//2
            self.decoder_blocks.append(
                ResBlock1D(curr_dim, curr_dim // 2, dropout=config.DROPOUT_RATE)
            )

            # Auxiliary Head Check
            if scale_factor in self.aux_scales:
                self.aux_indices.append(i)
                self.aux_heads.append(
                    nn.Conv1d(curr_dim // 2, self.out_channels, kernel_size=1)
                )

            curr_dim //= 2

        # Final Head
        self.final_head = nn.Conv1d(curr_dim, self.out_channels, kernel_size=1)

    def forward(self, x):
        # --- Encoder Path ---
        x = self.stem(x)

        skips = []
        for block, down in zip(self.encoder_blocks, self.downsamples):
            x = block(x)
            skips.append(x)
            x = down(x)

        # --- Bottleneck ---
        x = self.aspp(x)

        # --- Decoder Path ---
        outputs = {}

        # Iterate backwards through skips
        for i, (up, block) in enumerate(zip(self.upsamples, self.decoder_blocks)):
            skip = skips[-(i + 1)]

            x = up(x)

            # Handle padding issues if shapes don't match perfectly (due to odd lengths)
            if x.size(2) != skip.size(2):
                x = F.interpolate(
                    x, size=skip.size(2), mode="linear", align_corners=False
                )

            x = torch.cat([x, skip], dim=1)
            x = block(x)

            # Check if this level has an aux head
            if i in self.aux_indices:
                head_idx = self.aux_indices.index(i)
                scale = 2 ** (self.depth - 1 - i)
                outputs[f"aux_{scale}"] = self.aux_heads[head_idx](x)

        # Final output
        final_out = self.final_head(x)
        outputs["final"] = final_out

        return outputs
