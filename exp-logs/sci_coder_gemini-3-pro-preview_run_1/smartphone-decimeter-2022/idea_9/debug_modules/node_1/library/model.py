import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResBlock1D(nn.Module):
    """
    1D Residual Block with two convolution layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        super(ResBlock1D, self).__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            padding_mode="replicate",
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            padding_mode="replicate",
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.act2(out)
        return out


class BiGRUBottleneck(nn.Module):
    """
    Bi-directional GRU Bottleneck to capture long-term temporal dependencies.
    """

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.0):
        super(BiGRUBottleneck, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Project back to original channel dimension if needed,
        # or adapt decoder to take 2*hidden_dim
        self.out_proj = nn.Linear(hidden_dim * 2, input_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        b, c, t = x.size()

        # Permute for RNN: (Batch, Time, Channels)
        x_perm = x.permute(0, 2, 1)

        # GRU Output: (Batch, Time, Hidden*2)
        out, _ = self.gru(x_perm)

        # Project back: (Batch, Time, Channels)
        out = self.out_proj(out)
        out = self.act(out)

        # Permute back: (Batch, Channels, Time)
        out = out.permute(0, 2, 1)

        # Residual connection with input of bottleneck
        return out + x


class HybridResUNetGRU(nn.Module):
    """
    Hybrid 1D Residual U-Net with Bi-directional GRU Bottleneck.
    """

    def __init__(self):
        super(HybridResUNetGRU, self).__init__()

        # Calculate input dimension based on feature engineering config
        # CN0_BINS + ELEVATION_BINS + CN0_STATS(3) + ELEV_STATS(3) + SAT_COUNT(1) + UNC_MEAN(1)
        self.input_dim = Config.CN0_BINS + Config.ELEVATION_BINS + 3 + 3 + 1 + 1

        self.encoder_channels = Config.ENCODER_CHANNELS
        # We assume decoder channels are designed to match the upsampling path
        # Typically: [256, 128, 64, 32] for encoder [32, 64, 128, 256]

        # --- Encoder ---
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = self.input_dim
        for out_ch in self.encoder_channels:
            self.enc_blocks.append(
                ResBlock1D(
                    in_ch,
                    out_ch,
                    kernel_size=Config.KERNEL_SIZE,
                    dropout=Config.DROPOUT,
                )
            )
            # MaxPool with ceil_mode=True to handle odd lengths better
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True))
            in_ch = out_ch

        # --- Bottleneck ---
        # The bottleneck operates on the output of the last encoder block (before pooling? No, usually after)
        # Standard U-Net: Enc -> Pool -> Enc -> Pool ... -> Bottleneck -> Up
        # Here we put the GRU at the deepest level.
        self.bottleneck_conv = ResBlock1D(
            in_ch, in_ch * 2, kernel_size=Config.KERNEL_SIZE, dropout=Config.DROPOUT
        )
        self.gru_bottleneck = BiGRUBottleneck(
            input_dim=in_ch * 2,
            hidden_dim=Config.GRU_HIDDEN_DIM,
            num_layers=Config.GRU_LAYERS,
            dropout=Config.DROPOUT,
        )

        # --- Decoder ---
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        # Reverse encoder channels for decoder path
        # Bottleneck output is in_ch * 2
        current_ch = in_ch * 2

        # We iterate backwards through encoder channels to match skip connections
        skip_channels = self.encoder_channels[::-1]

        for i, skip_ch in enumerate(skip_channels):
            out_ch = skip_ch  # Target output channel for this decoder stage

            # Upsampling layer (Conv1d + Interpolate is often smoother than ConvTranspose1d)
            self.up_convs.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                    nn.Conv1d(
                        current_ch,
                        out_ch,
                        kernel_size=3,
                        padding=1,
                        padding_mode="replicate",
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )

            # Decoder block takes concatenated input (up_feat + skip_feat)
            self.dec_blocks.append(
                ResBlock1D(
                    out_ch + skip_ch,
                    out_ch,
                    kernel_size=Config.KERNEL_SIZE,
                    dropout=Config.DROPOUT,
                )
            )
            current_ch = out_ch

        # --- Output Head ---
        # Maps features to (North, East) offsets
        self.head = nn.Conv1d(current_ch, 2, kernel_size=1)

    def forward(self, x):
        # x: (Batch, Input_Dim, Time)

        skips = []

        # Encoder
        for block, pool in zip(self.enc_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck_conv(x)
        x = self.gru_bottleneck(x)

        # Decoder
        # Iterate over up_convs and dec_blocks
        # Skips need to be accessed in reverse order
        for i, (up, blk) in enumerate(zip(self.up_convs, self.dec_blocks)):
            skip = skips[-(i + 1)]

            x = up(x)

            # Handle size mismatch due to pooling of odd lengths
            if x.size(2) != skip.size(2):
                x = F.interpolate(
                    x, size=skip.size(2), mode="linear", align_corners=False
                )

            x = torch.cat([x, skip], dim=1)
            x = blk(x)

        # Output Head
        out = self.head(x)
        return out
