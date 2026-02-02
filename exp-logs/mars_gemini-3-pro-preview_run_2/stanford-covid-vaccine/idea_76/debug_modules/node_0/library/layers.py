import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class PermuteLayerNorm(nn.Module):
    """
    Applies LayerNorm to a tensor of shape (Batch, Channels, Length).
    PyTorch LayerNorm expects (Batch, ..., Channels).
    """

    def __init__(self, normalized_shape):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        # x: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)
        x = self.ln(x)
        # x: (B, L, C) -> (B, C, L)
        return x.transpose(1, 2)


class HybridInputStem(nn.Module):
    """
    Splits input into two branches:
    1. Branch A (Identity): Raw features (passed through).
    2. Branch B (Context): Spatial Conv -> LN -> SiLU.
    Concatenates them for the backbone.
    """

    def __init__(self, in_channels, context_channels):
        super().__init__()
        self.branch_b_conv = nn.Conv1d(
            in_channels,
            context_channels,
            kernel_size=Config.KERNEL_SIZE,
            padding=Config.KERNEL_SIZE // 2,
        )
        self.branch_b_norm = PermuteLayerNorm(context_channels)
        self.branch_b_act = nn.SiLU()

    def forward(self, x):
        # x: (B, C, L)

        # Branch A: Identity
        out_a = x

        # Branch B: Context
        out_b = self.branch_b_conv(x)
        out_b = self.branch_b_norm(out_b)
        out_b = self.branch_b_act(out_b)

        # Concatenate: (B, C_in + C_context, L)
        return torch.cat([out_a, out_b], dim=1)


class DenseDilatedBlock(nn.Module):
    """
    A Post-Activation block for the Dense Backbone.
    Structure: Dilated Conv (k=3) -> LN -> SiLU -> Pointwise Conv (k=1) -> LN -> SiLU -> Dropout.
    """

    def __init__(self, in_channels, growth_rate, dilation, dropout=0.1):
        super().__init__()

        # 1. Standard Dilated Conv (k=3)
        # Padding must handle dilation to maintain sequence length.
        # padding = dilation * (kernel_size - 1) / 2
        kernel_size = Config.KERNEL_SIZE
        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm1 = PermuteLayerNorm(growth_rate)
        self.act1 = nn.SiLU()

        # 2. Standard Pointwise Conv (k=1)
        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)
        self.norm2 = PermuteLayerNorm(growth_rate)
        self.act2 = nn.SiLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, in_channels, L)

        # Layer 1: Spatial Aggregation
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        # Layer 2: Channel Mixing
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        out = self.dropout(out)

        return out


class FeedbackStem(nn.Module):
    """
    Processes recycled predictions.
    1. Masks unscored channels (deg_pH10, deg_50C) to 0.
    2. Applies Conv -> LN -> SiLU.
    """

    def __init__(self, out_channels):
        super().__init__()
        self.unscored_indices = Config.UNSCORED_COLS_INDICES
        # Input is always 5 channels (Config.TARGET_COLS)
        self.in_channels = len(Config.TARGET_COLS)

        self.conv = nn.Conv1d(
            self.in_channels,
            out_channels,
            kernel_size=Config.KERNEL_SIZE,
            padding=Config.KERNEL_SIZE // 2,
        )
        self.norm = PermuteLayerNorm(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, L, 5) - Predictions usually come in (B, L, C) format from the model output

        # Transpose to (B, C, L) for Conv1d
        x = x.transpose(1, 2)

        # Strict Channel Masking
        # Create a mask of ones
        mask = torch.ones_like(x)
        # Zero out the specific unscored channels
        mask[:, self.unscored_indices, :] = 0.0

        # Apply mask
        x_masked = x * mask

        # Process
        out = self.conv(x_masked)
        out = self.norm(out)
        out = self.act(out)

        # Output: (B, out_channels, L)
        return out
