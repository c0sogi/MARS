import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbeddingStem(nn.Module):
    """
    Projects categorical inputs (e.g., One-Hot encoded) into a dense latent space
    using a Pointwise Convolution (Kernel Size = 1).

    This is strictly required before Pre-Activation blocks to avoid signal destruction
    of sparse/discrete inputs by the first Normalization layer.
    """

    def __init__(self, in_channels, out_channels):
        super(InputEmbeddingStem, self).__init__()
        self.stem = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, L) or (B, L, C).
               If (B, L, C), it will be permuted to (B, C, L).
        Returns:
            Tensor of shape (B, out_channels, L).
        """
        # Ensure input is (B, C, L) for Conv1d
        # Check if the second dimension matches in_channels, if not, assume (B, L, C) and permute
        if (
            x.dim() == 3
            and x.shape[1] != self.stem.in_channels
            and x.shape[2] == self.stem.in_channels
        ):
            x = x.permute(0, 2, 1)

        return self.stem(x)


class PreActDilatedBlock(nn.Module):
    """
    Micro-Architecture:
    LayerNorm -> SiLU -> Dilated Conv (k=3) -> LayerNorm -> SiLU -> Pointwise Conv (k=1) -> Dropout

    Designed for DenseNets:
    - Input channels = Accumulation of previous layers
    - Output channels = Growth Rate
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(PreActDilatedBlock, self).__init__()

        # 1. First Unit: LN -> SiLU -> Dilated Conv (Spatial Mixing)
        self.ln1 = nn.LayerNorm(in_channels)
        self.act1 = nn.SiLU()
        # Note: In this specific architecture description, the spatial conv (k=3) comes first.
        # We project to 'out_channels' (growth rate) immediately to manage parameter count
        # if in_channels is large (due to dense connections).
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=dilation,
            dilation=dilation,
        )

        # 2. Second Unit: LN -> SiLU -> Pointwise Conv (Channel Mixing / Projection)
        self.ln2 = nn.LayerNorm(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C_in, L)

        # --- Unit 1 ---
        # LayerNorm expects (B, L, C), so we permute
        out = x.permute(0, 2, 1)
        out = self.ln1(out)
        out = self.act1(out)
        out = out.permute(0, 2, 1)

        out = self.conv1(out)

        # --- Unit 2 ---
        out = out.permute(0, 2, 1)
        out = self.ln2(out)
        out = self.act2(out)
        out = out.permute(0, 2, 1)

        out = self.conv2(out)
        out = self.dropout(out)

        return out


class DenseTCN(nn.Module):
    """
    A stack of Single-Layer Dilated Blocks utilizing Dense Connections.

    Mechanism:
    - Iterates through the specified dilation rates.
    - At each step, concatenates the outputs of ALL prior blocks (including input)
      to form the input for the current block.
    - Dynamically tracks the increasing input channel depth.
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilations, dropout):
        super(DenseTCN, self).__init__()
        self.blocks = nn.ModuleList()
        self.dilations = dilations

        current_in_channels = in_channels

        for d in dilations:
            block = PreActDilatedBlock(
                in_channels=current_in_channels,
                out_channels=growth_rate,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
            )
            self.blocks.append(block)

            # In a DenseNet, the input to the next layer grows by the output of the current layer
            current_in_channels += growth_rate

        self.out_channels = current_in_channels

    def forward(self, x):
        # x: (B, C, L)

        # List to store feature maps for concatenation
        features = [x]

        for block in self.blocks:
            # Dense Connection: Concatenate all previous features along channel dim
            inp = torch.cat(features, dim=1)

            # Compute block output
            out = block(inp)

            # Add output to features list for subsequent layers
            features.append(out)

        # Return the final dense concatenation of all features
        return torch.cat(features, dim=1)
