import torch
import torch.nn as nn
from library.config import Config


class DilatedConvBlock(nn.Module):
    """
    A residual block containing a dilated 1D convolution, Batch Normalization,
    ReLU activation, and Dropout.

    The padding is dynamically calculated based on dilation and kernel size
    to maintain the sequence length (same padding).
    """

    def __init__(self, channels, dilation, kernel_size, dropout):
        super(DilatedConvBlock, self).__init__()

        # Calculate padding to maintain sequence length
        # Formula for same padding with stride 1: P = dilation * (kernel_size - 1) / 2
        # We assume kernel_size is odd (e.g., 3).
        self.padding = dilation * (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
            bias=False,  # Bias is redundant with BatchNorm
        )

        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out + residual


class RNAResNet(nn.Module):
    """
    Dilated Residual Convolutional Network for RNA degradation prediction.

    Architecture:
    1. Input Projection (Conv1d k=1)
    2. Stack of Dilated Residual Blocks (increasing dilation)
    3. Output Projection (Conv1d k=1)
    """

    def __init__(
        self,
        input_channels=Config.INPUT_CHANNELS,
        num_targets=Config.NUM_TARGETS,
        hidden_dim=Config.HIDDEN_DIM,
        kernel_size=Config.KERNEL_SIZE,
        dropout=Config.DROPOUT,
        dilations=Config.DILATIONS,
    ):
        super(RNAResNet, self).__init__()

        # 1. Initial Projection: Map input features to hidden dimension
        self.embedding = nn.Conv1d(
            in_channels=input_channels, out_channels=hidden_dim, kernel_size=1
        )

        # 2. Backbone: Stack of Dilated Residual Blocks
        self.blocks = nn.ModuleList()
        for dilation in dilations:
            self.blocks.append(
                DilatedConvBlock(
                    channels=hidden_dim,
                    dilation=dilation,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )

        # 3. BiGRU for global context (Cite solution_lesson_node_00003)
        # Optimization: Set hidden_size to half of input_dim to maintain dimension consistency (Cite solution_lesson_node_00004)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        # 4. Output Head: Map GRU output (hidden_dim) to target predictions
        self.head = nn.Conv1d(
            in_channels=hidden_dim, out_channels=num_targets, kernel_size=1
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Input_Channels).

        Returns:
            torch.Tensor: Output tensor of shape (Batch, Seq_Len, Num_Targets).
        """
        # Permute input to (Batch, Channels, Seq_Len) for Conv1d
        x = x.permute(0, 2, 1)

        # Initial embedding
        x = self.embedding(x)

        # Pass through dilated residual blocks
        for block in self.blocks:
            x = block(x)

        # Permute to (Batch, Seq_Len, Hidden_Dim) for GRU
        x = x.permute(0, 2, 1)

        # Pass through BiGRU (Cite solution_lesson_node_00003)
        # Output shape: (Batch, Seq_Len, 2 * Hidden_Dim)
        x, _ = self.gru(x)

        # Permute back to (Batch, Channels, Seq_Len) for Conv1d head
        x = x.permute(0, 2, 1)

        # Final projection
        x = self.head(x)

        # Permute output back to (Batch, Seq_Len, Targets)
        x = x.permute(0, 2, 1)

        return x
