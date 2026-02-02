import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with two convolution layers, Batch Normalization, and ReLU.
    Maintains temporal resolution.
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

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (1D) to capture multi-scale temporal context.
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

        # Final projection
        # Input channels = (len(dilations) + 1 + 1) * out_channels
        self.project = nn.Sequential(
            nn.Conv1d((len(dilations) + 2) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.modules_list:
            res.append(conv(x))

        # Global pooling branch
        gap = self.global_avg_pool(x)
        gap = F.interpolate(gap, size=x.size(2), mode="nearest")
        res.append(gap)

        res = torch.cat(res, dim=1)
        return self.project(res)


class AttentionGate(nn.Module):
    """
    Attention Gate to filter encoder features based on decoder features.

    Args:
        f_g (int): Number of channels in the gating signal (decoder feature).
        f_l (int): Number of channels in the skip connection (encoder feature).
        f_int (int): Number of intermediate channels.
    """

    def __init__(self, f_g, f_l, f_int):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv1d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(f_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(f_int),
        )

        self.psi = nn.Sequential(
            nn.Conv1d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: Gating signal from decoder (upsampled).
        x: Skip connection from encoder.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Add and ReLU
        psi = self.relu(g1 + x1)

        # Compute attention coefficients (0 to 1)
        alpha = self.psi(psi)

        # Weight the encoder features
        return x * alpha


class PhaseAwareStratified1DAttentionResUNet(nn.Module):
    def __init__(self, config=None):
        super(PhaseAwareStratified1DAttentionResUNet, self).__init__()

        if config is None:
            config = Config()

        self.config = config

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool1d(2)

        in_ch = config.INPUT_CHANNELS

        for out_ch in config.ENCODER_FILTERS:
            self.encoder_blocks.append(ResidualBlock1D(in_ch, out_ch))
            in_ch = out_ch

        # Bottleneck (ASPP)
        # Input to ASPP is the output of the last encoder block
        aspp_in_ch = config.ENCODER_FILTERS[-1]
        # We set ASPP out to match last encoder filter count for compatibility
        self.aspp = ASPP(aspp_in_ch, aspp_in_ch, config.ASPP_DILATIONS)

        # Decoder
        self.up_convs = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.aux_heads = nn.ModuleList()

        # Reverse encoder filters for skips
        skip_channels = config.ENCODER_FILTERS[::-1]
        decoder_filters = config.DECODER_FILTERS

        # Current channel count coming from bottleneck
        curr_ch = aspp_in_ch

        for i, out_ch in enumerate(decoder_filters):
            skip_ch = skip_channels[i]

            # Upsampling (Transpose Conv)
            self.up_convs.append(
                nn.ConvTranspose1d(curr_ch, out_ch, kernel_size=2, stride=2)
            )

            # Attention Gate
            # g = upsampled feature (out_ch), x = skip (skip_ch)
            if config.USE_ATTENTION_GATES:
                self.att_gates.append(
                    AttentionGate(f_g=out_ch, f_l=skip_ch, f_int=out_ch // 2)
                )
            else:
                self.att_gates.append(nn.Identity())

            # Decoder Block
            # Input to block is cat(upsampled, gated_skip) -> out_ch + skip_ch
            self.decoder_blocks.append(ResidualBlock1D(out_ch + skip_ch, out_ch))

            # Auxiliary Heads (Deep Supervision)
            if config.USE_DEEP_SUPERVISION and i < len(decoder_filters) - 1:
                self.aux_heads.append(
                    nn.Conv1d(out_ch, config.OUTPUT_CHANNELS, kernel_size=1)
                )
            else:
                self.aux_heads.append(None)

            curr_ch = out_ch

        # Final Output Head
        self.final_conv = nn.Conv1d(curr_ch, config.OUTPUT_CHANNELS, kernel_size=1)

    def forward(self, x):
        # x: [Batch, Channels, Length]

        # Encoder
        skips = []
        for block in self.encoder_blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.aspp(x)

        # Decoder
        # Skips need to be accessed in reverse order
        skips = skips[::-1]

        aux_outputs = []

        for i in range(len(self.decoder_blocks)):
            # Upsample
            x = self.up_convs[i](x)

            # Get skip connection
            skip = skips[i]

            # Handle shape mismatch due to padding in dataset
            if x.shape[2] != skip.shape[2]:
                x = F.interpolate(
                    x, size=skip.shape[2], mode="linear", align_corners=True
                )

            # Attention Gate
            if self.config.USE_ATTENTION_GATES:
                gated_skip = self.att_gates[i](g=x, x=skip)
            else:
                gated_skip = skip

            # Concatenate
            x = torch.cat([x, gated_skip], dim=1)

            # Residual Block
            x = self.decoder_blocks[i](x)

            # Deep Supervision
            if self.config.USE_DEEP_SUPERVISION and self.aux_heads[i] is not None:
                aux_outputs.append(self.aux_heads[i](x))

        # Final Output
        final_out = self.final_conv(x)

        if self.training and self.config.USE_DEEP_SUPERVISION:
            return final_out, aux_outputs
        else:
            return final_out
