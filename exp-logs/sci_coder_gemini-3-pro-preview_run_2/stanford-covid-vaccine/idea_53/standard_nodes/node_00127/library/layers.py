import torch
import torch.nn as nn


class SpatialInputStem(nn.Module):
    """
    Spatial Input Stem:
    Processes input features with a Spatial Convolution (k=3) -> LayerNorm -> SiLU.
    This ensures immediate mixing of adjacent nucleotides (n-grams) creating a dense embedding
    before the first normalization step, as per Lesson 00125.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        # Padding to maintain sequence length
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: [Batch, In_Channels, Seq_Len]
        x = self.conv(x)

        # LayerNorm expects [Batch, Seq_Len, Channels]
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)

        # Output: [Batch, Out_Channels, Seq_Len]
        return x


class PostActDenseBlock(nn.Module):
    """
    Post-Activation Dense Block:
    Decouples spatial aggregation from channel mixing.
    Structure: Dilated Conv (k=3) -> LN -> SiLU -> Pointwise Conv (k=1) -> LN -> SiLU -> Dropout.
    """

    def __init__(
        self, in_channels, growth_rate, kernel_size=3, dilation=1, dropout=0.1
    ):
        super().__init__()
        padding = dilation * (kernel_size // 2)

        # 1. Spatial Aggregation (Dilated Conv)
        self.conv_spatial = nn.Conv1d(
            in_channels, growth_rate, kernel_size, padding=padding, dilation=dilation
        )
        self.norm1 = nn.LayerNorm(growth_rate)
        self.act1 = nn.SiLU()

        # 2. Channel Mixing (Pointwise Conv)
        self.conv_mixing = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)
        self.norm2 = nn.LayerNorm(growth_rate)
        self.act2 = nn.SiLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, In_Channels, Seq_Len] (Concatenated history)

        # Spatial Conv
        out = self.conv_spatial(x)

        # LN + SiLU
        out = out.permute(0, 2, 1)
        out = self.norm1(out)
        out = self.act1(out)
        out = out.permute(0, 2, 1)

        # Mixing Conv
        out = self.conv_mixing(out)

        # LN + SiLU + Dropout
        out = out.permute(0, 2, 1)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)

        # Output: [Batch, Growth_Rate, Seq_Len]
        return out


class DenseTCN(nn.Module):
    """
    Dense Dilated TCN:
    Manages a stack of PostActDenseBlocks with dense connections.
    The input to each block is the concatenation of inputs and all previous block outputs.
    """

    def __init__(self, in_channels, growth_rate, dilations, kernel_size=3, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.dilations = dilations

        current_in_channels = in_channels

        for d in dilations:
            block = PostActDenseBlock(
                in_channels=current_in_channels,
                growth_rate=growth_rate,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
            )
            self.blocks.append(block)
            # In a DenseNet, the next block sees all previous channels + new growth_rate
            current_in_channels += growth_rate

        self.out_channels = current_in_channels

    def forward(self, x):
        # x: [Batch, In_Channels, Seq_Len]
        features = [x]

        for block in self.blocks:
            # Dense connection: concatenate all previous features
            inp = torch.cat(features, dim=1)
            out = block(inp)
            features.append(out)

        # Return the full dense state (concatenation of input and all block outputs)
        return torch.cat(features, dim=1)
