import torch
import torch.nn as nn


class RobustBlock(nn.Module):
    """
    A robust pre-activation block for RNA sequence modeling.

    Structure:
    LayerNorm -> SiLU -> Dilated Conv (k=3) -> LayerNorm -> SiLU -> Pointwise Conv (k=1) -> Dropout

    This block decouples spatial aggregation (dilated conv) from channel mixing (pointwise conv)
    and uses pre-activation to improve gradient flow.
    """

    def __init__(self, in_channels, out_channels, dilation, dropout=0.1):
        super().__init__()

        # Pre-activation 1: Spatial Aggregation
        self.ln1 = nn.LayerNorm(in_channels)
        self.act1 = nn.SiLU()
        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

        # Pre-activation 2: Channel Mixing
        self.ln2 = nn.LayerNorm(out_channels)
        self.act2 = nn.SiLU()
        self.conv_pointwise = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Channels, Length)
        Returns:
            Output tensor of shape (Batch, Out_Channels, Length)
        """
        # LN expects (B, L, C), Conv expects (B, C, L)

        # 1. Spatial Aggregation Path
        out = x.transpose(1, 2)  # (B, L, C)
        out = self.ln1(out)
        out = self.act1(out)
        out = out.transpose(1, 2)  # (B, C, L)

        out = self.conv_dilated(out)

        # 2. Channel Mixing Path
        out_ln = out.transpose(1, 2)
        out_ln = self.ln2(out_ln)
        out_ln = self.act2(out_ln)
        out_ln = out_ln.transpose(1, 2)

        out = self.conv_pointwise(out_ln)
        out = self.drop(out)

        return out


class DenseDilatedTCN(nn.Module):
    """
    A stack of RobustBlocks with dense connections.

    Each block receives the concatenation of the original input and all previous blocks' outputs.
    This facilitates feature reuse and gradient flow across deep networks.
    """

    def __init__(self, in_channels, growth_rate, dilations, dropout=0.1):
        """
        Args:
            in_channels: Number of channels in the input tensor.
            growth_rate: Number of output channels for each block.
            dilations: List of integers specifying dilation rate for each block.
            dropout: Dropout probability.
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        current_dim = in_channels

        for d in dilations:
            # Input dim grows by growth_rate at each step due to dense concatenation
            self.blocks.append(
                RobustBlock(current_dim, growth_rate, dilation=d, dropout=dropout)
            )
            current_dim += growth_rate

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, In_Channels, Length)
        Returns:
            Concatenated tensor of shape (Batch, In_Channels + Num_Blocks * Growth_Rate, Length)
        """
        features = [x]

        for block in self.blocks:
            # Dense connection: concatenate all prior features
            inp = torch.cat(features, dim=1)
            out = block(inp)
            features.append(out)

        # Return the dense state (concatenation of all features)
        return torch.cat(features, dim=1)
