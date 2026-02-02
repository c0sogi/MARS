import torch
import torch.nn as nn


class DecoupledDenseBlock(nn.Module):
    """
    Implements the Decoupled Block Micro-Architecture.
    Structure: Norm -> ReLU -> Dilated Conv (Depthwise) -> Norm -> ReLU -> Pointwise Conv.

    This separates spatial aggregation (via depthwise dilated convolution) from
    channel mixing (via pointwise convolution), ensuring parameter efficiency
    and robust feature extraction within a DenseNet-style architecture.
    """

    def __init__(
        self, in_channels, growth_rate, kernel_size=3, dilation=1, dropout=0.0
    ):
        super(DecoupledDenseBlock, self).__init__()

        # 1. Spatial Aggregation: Depthwise Dilated Convolution
        # Maps in_channels -> in_channels (preserving depth, aggregating spatial context)
        self.norm1 = nn.BatchNorm1d(in_channels)
        self.act1 = nn.ReLU()
        self.conv_spatial = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation,  # Maintain sequence length (assuming k=3, odd kernel)
            groups=in_channels,  # Depthwise
            bias=False,
        )

        # 2. Channel Mixing: Pointwise Convolution
        # Maps in_channels -> growth_rate (generating new features)
        self.norm2 = nn.BatchNorm1d(in_channels)
        self.act2 = nn.ReLU()
        self.conv_pointwise = nn.Conv1d(
            in_channels, growth_rate, kernel_size=1, bias=False
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, in_channels, Seq_Len)

        Returns:
            torch.Tensor: New features of shape (Batch, growth_rate, Seq_Len)
        """
        # Part 1: Spatial
        out = self.norm1(x)
        out = self.act1(out)
        out = self.conv_spatial(out)

        # Part 2: Channel
        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv_pointwise(out)

        out = self.dropout(out)
        return out


class DenseTCNStack(nn.Module):
    """
    A stack of DecoupledDenseBlocks with Dense Connectivity.

    Features from all preceding layers are concatenated to form the input
    for the next layer. This maximizes information flow and feature reuse.
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilations, dropout=0.0):
        super(DenseTCNStack, self).__init__()
        self.blocks = nn.ModuleList()
        self.in_channels = in_channels

        current_channels = in_channels

        for d in dilations:
            block = DecoupledDenseBlock(
                in_channels=current_channels,
                growth_rate=growth_rate,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
            )
            self.blocks.append(block)

            # In DenseNet, the input to the next layer grows by growth_rate
            current_channels += growth_rate

        self.out_channels = current_channels

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, in_channels, Seq_Len)

        Returns:
            torch.Tensor: Concatenated features of shape (Batch, out_channels, Seq_Len)
        """
        # Initialize feature list with the input
        features = [x]

        for block in self.blocks:
            # Dense Connection: Concatenate all previous features
            # Input to block i has channels = in_channels + i * growth_rate
            in_tensor = torch.cat(features, dim=1)

            # Block produces new features of size growth_rate
            new_features = block(in_tensor)

            # Add new features to the stack
            features.append(new_features)

        # Return the full stack of features
        return torch.cat(features, dim=1)
