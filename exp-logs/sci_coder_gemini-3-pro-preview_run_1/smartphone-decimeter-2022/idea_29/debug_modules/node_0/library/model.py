import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResBlock1D(nn.Module):
    """
    Basic 1D Residual Block: Conv1d -> BN -> ReLU -> Dropout -> Conv1d -> BN -> Residual Add -> ReLU
    """

    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.act(out)
        return out


class FusionUnit(nn.Module):
    """
    Multi-Scale Fusion Unit.
    Exchanges information between parallel streams of different resolutions.
    """

    def __init__(self, num_streams, stream_channels, resolution_factors):
        super().__init__()
        self.num_streams = num_streams
        self.stream_channels = stream_channels
        self.resolution_factors = resolution_factors

        # Create transformation layers for every pair (src, dst)
        self.transforms = nn.ModuleDict()

        for src in range(num_streams):
            for dst in range(num_streams):
                if src == dst:
                    continue

                src_c = stream_channels[src]
                dst_c = stream_channels[dst]
                src_res = resolution_factors[src]
                dst_res = resolution_factors[dst]

                key = f"{src}_{dst}"

                if src_res < dst_res:
                    # Downsample (High Res -> Low Res)
                    # Factor = dst_res / src_res (e.g. 4/1 = 4)
                    stride = int(dst_res / src_res)
                    # Use a strided convolution
                    self.transforms[key] = nn.Sequential(
                        nn.Conv1d(
                            src_c, dst_c, kernel_size=stride, stride=stride, bias=False
                        ),
                        nn.BatchNorm1d(dst_c),
                    )
                else:
                    # Upsample (Low Res -> High Res)
                    # 1x1 Conv to match channels, then interpolate in forward
                    self.transforms[key] = nn.Sequential(
                        nn.Conv1d(src_c, dst_c, kernel_size=1, bias=False),
                        nn.BatchNorm1d(dst_c),
                    )

    def forward(self, streams):
        # streams: list of tensors [s0, s1, s2] corresponding to resolutions
        new_streams = []

        for dst in range(self.num_streams):
            # Start with identity (if we considered it a residual fusion, but HRNet usually sums transformed inputs)
            # Here we sum all contributions.
            out = streams[dst]

            for src in range(self.num_streams):
                if src == dst:
                    continue

                key = f"{src}_{dst}"
                layer = self.transforms[key]
                x = streams[src]

                if self.resolution_factors[src] < self.resolution_factors[dst]:
                    # Downsample
                    # Pad x to be divisible by stride if necessary (though global padding usually handles this)
                    feat = layer(x)

                    # Ensure size match (crop or pad if slight mismatch due to rounding)
                    target_len = streams[dst].shape[-1]
                    if feat.shape[-1] > target_len:
                        feat = feat[..., :target_len]
                    elif feat.shape[-1] < target_len:
                        feat = F.pad(feat, (0, target_len - feat.shape[-1]))

                else:
                    # Upsample
                    # Apply 1x1 conv first
                    feat = layer(x)
                    # Then interpolate to target size
                    target_len = streams[dst].shape[-1]
                    feat = F.interpolate(
                        feat, size=target_len, mode="linear", align_corners=False
                    )

                out = out + feat

            new_streams.append(F.relu(out))  # Apply ReLU after fusion sum

        return new_streams


class HR1DResNet(nn.Module):
    """
    High-Resolution 1D Residual Network with Parallel Multi-Resolution Streams.
    """

    def __init__(self):
        super().__init__()

        # Config
        self.in_channels = Config.IN_CHANNELS
        self.stem_channels = Config.STEM_CHANNELS
        self.stream_channels = Config.STREAM_CHANNELS
        self.res_factors = Config.RESOLUTION_FACTORS
        self.num_stages = Config.NUM_STAGES
        self.blocks_per_stage = Config.BLOCKS_PER_STAGE
        self.out_dim = Config.OUT_DIM
        self.dropout = Config.DROPOUT

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.in_channels,
                self.stem_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(self.stem_channels),
            nn.ReLU(inplace=True),
        )

        # Initial Transitions to create parallel streams from the stem
        self.init_transforms = nn.ModuleList()
        for i, ch in enumerate(self.stream_channels):
            factor = self.res_factors[i]
            if i == 0:
                # Stream 0 (High Res)
                if ch == self.stem_channels:
                    self.init_transforms.append(nn.Identity())
                else:
                    self.init_transforms.append(nn.Conv1d(self.stem_channels, ch, 1))
            else:
                # Lower Res Streams: Downsample from stem
                self.init_transforms.append(
                    nn.Sequential(
                        nn.Conv1d(
                            self.stem_channels,
                            ch,
                            kernel_size=factor,
                            stride=factor,
                            bias=False,
                        ),
                        nn.BatchNorm1d(ch),
                        nn.ReLU(inplace=True),
                    )
                )

        # Stages and Fusions
        self.stages = nn.ModuleList()
        self.fusions = nn.ModuleList()

        for stage_idx in range(self.num_stages):
            # Blocks for each stream
            stage_blocks = nn.ModuleList()
            for stream_idx in range(len(self.stream_channels)):
                ch = self.stream_channels[stream_idx]
                blocks = nn.Sequential(
                    *[
                        ResBlock1D(ch, dropout=self.dropout)
                        for _ in range(self.blocks_per_stage)
                    ]
                )
                stage_blocks.append(blocks)
            self.stages.append(stage_blocks)

            # Fusion Unit after each stage
            self.fusions.append(
                FusionUnit(
                    len(self.stream_channels), self.stream_channels, self.res_factors
                )
            )

        # Output Heads (Deep Supervision)
        self.heads = nn.ModuleList()
        for ch in self.stream_channels:
            self.heads.append(nn.Conv1d(ch, self.out_dim, kernel_size=1))

    def forward(self, x):
        # x: [B, C, T]

        # Pad input T to be divisible by max resolution factor (16)
        # This ensures downsampling/upsampling logic works smoothly
        max_factor = max(self.res_factors)
        orig_len = x.shape[-1]
        pad_len = (max_factor - (orig_len % max_factor)) % max_factor
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        # Apply Stem
        x = self.stem(x)

        # Initialize Parallel Streams
        streams = []
        for trans in self.init_transforms:
            streams.append(trans(x))

        # Process Stages
        for stage_blocks, fusion in zip(self.stages, self.fusions):
            # Apply blocks to each stream independently
            new_streams_after_blocks = []
            for i, block in enumerate(stage_blocks):
                new_streams_after_blocks.append(block(streams[i]))

            # Apply fusion to mix information
            streams = fusion(new_streams_after_blocks)

        # Generate Outputs from all heads
        outputs = []
        for i, head in enumerate(self.heads):
            out = head(streams[i])

            # For the High-Res stream (index 0), crop back to original length
            if i == 0:
                out = out[..., :orig_len]

            outputs.append(out)

        return outputs
