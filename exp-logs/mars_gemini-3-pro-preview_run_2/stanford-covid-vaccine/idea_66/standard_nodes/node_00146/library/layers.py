import torch
import torch.nn as nn
from library.config import Config


class SpatialStem(nn.Module):
    """
    Spatial Input Stem: Standard Convolution (k=3) -> LayerNorm -> SiLU.
    Ensures immediate mixing of adjacent nucleotides to create a dense, context-rich embedding.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.ln = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, C_in, L)
        x = self.conv(x)

        # LayerNorm expects (B, L, C), so permute
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = self.act(x)

        # Permute back to (B, C, L)
        x = x.permute(0, 2, 1)
        return x


class DenseDilatedBlock(nn.Module):
    """
    Single-Layer Dilated Block utilizing Post-Activation structure.
    Structure: LayerNorm -> SiLU -> Dilated Conv (k=3) -> LayerNorm -> SiLU -> Pointwise Conv (k=1) -> Dropout.
    """

    def __init__(self, in_channels, growth_rate, dilation, dropout=Config.DROPOUT):
        super().__init__()

        # 1. Pre-activation for Dilated Conv
        self.ln1 = nn.LayerNorm(in_channels)
        self.act1 = nn.SiLU()

        # Standard Dilated Conv (k=3)
        # Note: Input channels = accumulated channels from dense connections
        self.conv_dilated = nn.Conv1d(
            in_channels, growth_rate, kernel_size=3, padding=dilation, dilation=dilation
        )

        # 2. Post-processing for Pointwise Conv
        self.ln2 = nn.LayerNorm(growth_rate)
        self.act2 = nn.SiLU()

        # Standard Pointwise Conv (k=1)
        self.conv_point = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C_in, L)

        # Branch start: Post-Activation logic
        out = x.permute(0, 2, 1)  # (B, L, C)
        out = self.ln1(out)
        out = self.act1(out)
        out = out.permute(0, 2, 1)  # (B, C, L)

        out = self.conv_dilated(out)

        out = out.permute(0, 2, 1)  # (B, L, Growth)
        out = self.ln2(out)
        out = self.act2(out)
        out = out.permute(0, 2, 1)  # (B, Growth, L)

        out = self.conv_point(out)
        out = self.dropout(out)

        return out


class DenseTCN(nn.Module):
    """
    Dense Dilated TCN Backbone.
    Manages a stack of DenseDilatedBlocks with dense connections (concatenating outputs of all prior blocks).
    """

    def __init__(
        self,
        in_channels,
        growth_rate=Config.HIDDEN_DIM,
        dilations=Config.DILATIONS,
        dropout=Config.DROPOUT,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.growth_rate = growth_rate

        current_dim = in_channels

        for d in dilations:
            # Each block takes the concatenation of all previous features
            block = DenseDilatedBlock(
                in_channels=current_dim,
                growth_rate=growth_rate,
                dilation=d,
                dropout=dropout,
            )
            self.blocks.append(block)
            current_dim += growth_rate

        self.out_channels = current_dim

    def forward(self, x):
        # x: (B, C_in, L)
        features = [x]

        for block in self.blocks:
            # Dense connection: Concatenate all prior features
            in_feat = torch.cat(features, dim=1)
            new_feat = block(in_feat)
            features.append(new_feat)

        # Return concatenation of all features (DenseNet style)
        return torch.cat(features, dim=1)
