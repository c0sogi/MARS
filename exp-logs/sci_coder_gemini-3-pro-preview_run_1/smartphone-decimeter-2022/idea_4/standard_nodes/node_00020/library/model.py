import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class EncoderBlock(nn.Module):
    """
    Encoder block for 1D U-Net.
    Consists of two 1D convolutions, batch normalization, ReLU activation,
    dropout, and max pooling.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        skip = x  # Store feature map for skip connection
        x = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    """
    Decoder block for 1D U-Net.
    Upsamples the input, concatenates with the skip connection, and applies convolutions.
    Handles dimension mismatch caused by odd input lengths during pooling.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        # Input channels to conv is sum of upsampled features and skip connection features
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle padding if shapes don't match due to odd sequence lengths
        if x.shape[2] != skip.shape[2]:
            diff = skip.shape[2] - x.shape[2]
            # Pad the last dimension (time)
            # F.pad format for 3D input (N, C, L) is (padding_left, padding_right)
            x = F.pad(x, (diff // 2, diff - diff // 2))

        # Concatenate along channel dimension
        x = torch.cat([x, skip], dim=1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class UNet1D(nn.Module):
    """
    1D U-Net architecture for time-series regression.
    """

    def __init__(
        self,
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUTPUT_CHANNELS,
        base_filters=Config.BASE_FILTERS,
        depth=Config.DEPTH,
        kernel_size=Config.KERNEL_SIZE,
        dropout=Config.DROPOUT,
    ):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # --- Encoder Path ---
        curr_filters = in_channels
        next_filters = base_filters

        for _ in range(depth):
            self.encoders.append(
                EncoderBlock(curr_filters, next_filters, kernel_size, dropout)
            )
            curr_filters = next_filters
            next_filters *= 2

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv1d(
                curr_filters, next_filters, kernel_size, padding=kernel_size // 2
            ),
            nn.BatchNorm1d(next_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                next_filters, next_filters, kernel_size, padding=kernel_size // 2
            ),
            nn.BatchNorm1d(next_filters),
            nn.ReLU(),
        )
        curr_filters = next_filters

        # --- Decoder Path ---
        # The decoder mirrors the encoder depth
        for _ in range(depth):
            # Target output filters for this block (halving the bottleneck/previous layer)
            out_filters = curr_filters // 2

            # The input to the decoder conv is the concatenation of:
            # 1. Upsampled previous layer (size: curr_filters)
            # 2. Skip connection from encoder (size: out_filters)
            # Total input channels = curr_filters + out_filters
            self.decoders.append(
                DecoderBlock(
                    curr_filters + out_filters, out_filters, kernel_size, dropout
                )
            )

            curr_filters = out_filters

        # --- Final Output Layer ---
        self.final_conv = nn.Conv1d(curr_filters, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, In_Channels, Time)
        Returns:
            out: Output tensor of shape (Batch, Out_Channels, Time)
        """
        skips = []

        # Encoder Pass
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder Pass
        # Iterate decoders and skip connections in reverse order
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        # Final Prediction
        out = self.final_conv(x)

        return out
