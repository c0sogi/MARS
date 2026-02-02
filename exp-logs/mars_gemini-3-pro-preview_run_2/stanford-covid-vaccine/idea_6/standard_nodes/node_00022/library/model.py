import torch
import torch.nn as nn
from library.config import Config


class GatedBlock(nn.Module):
    """
    A Gated Dilated Convolutional Block with Residual Connection.

    Mechanism:
        z = tanh(W_f * x) * sigmoid(W_g * x)
        out = Dropout(z) + Residual(x)
    """

    def __init__(self, in_channels, out_channels, dilation, kernel_size, dropout):
        super(GatedBlock, self).__init__()
        # Calculate padding to maintain sequence length (assuming 'same' padding logic)
        # padding = (kernel_size - 1) * dilation // 2
        padding = (kernel_size - 1) * dilation // 2

        self.conv_filter = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv_gate = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)

        # Residual connection: Use 1x1 conv if channel dimensions change, else Identity
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # x: [Batch, Channels, Seq_Len]

        # Filter (Feature extraction)
        filter_out = torch.tanh(self.conv_filter(x))

        # Gate (Information flow control)
        gate_out = torch.sigmoid(self.conv_gate(x))

        # Gated activation
        out = filter_out * gate_out
        out = self.dropout(out)

        # Residual addition
        res = self.residual(x)
        return out + res


class StackingAwareHybridNet(nn.Module):
    """
    Stacking-Aware Gated Hybrid Network.

    Architecture:
    1. Linear Embedding
    2. Gated Dilated TCN Backbone (Exponentially increasing dilation)
    3. Bidirectional GRU (Global Aggregation)
    4. Linear Output Head
    """

    def __init__(self, config=Config):
        super(StackingAwareHybridNet, self).__init__()

        self.input_dim = config.INPUT_DIM
        self.hidden_dim = config.HIDDEN_DIM
        self.output_dim = config.OUTPUT_DIM
        self.num_layers = config.NUM_LAYERS
        self.kernel_size = config.KERNEL_SIZE
        self.dropout = config.DROPOUT

        # 1. Input Embedding
        # Projects input features (Sequence + Structure + Loop + Partner Triplet) to Hidden Dim
        self.embedding = nn.Linear(self.input_dim, self.hidden_dim)

        # 2. Gated Dilated TCN Backbone
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            dilation = 2**i
            self.blocks.append(
                GatedBlock(
                    self.hidden_dim,
                    self.hidden_dim,
                    dilation,
                    self.kernel_size,
                    self.dropout,
                )
            )

        # 3. Global Aggregation (BiGRU)
        # We set hidden_size to hidden_dim // 2 so that the concatenated bidirectional output
        # has size hidden_dim, matching the TCN output dimension.
        self.bigru = nn.GRU(
            self.hidden_dim,
            self.hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 4. Output Head
        self.head = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # x: [Batch, Seq_Len, Input_Dim]

        # Embedding
        x = self.embedding(x)  # [Batch, Seq, Hidden]

        # Permute for Conv1d: [Batch, Hidden, Seq]
        x = x.permute(0, 2, 1)

        # Pass through TCN blocks
        for block in self.blocks:
            x = block(x)

        # Permute back for GRU: [Batch, Seq, Hidden]
        x = x.permute(0, 2, 1)

        # BiGRU Aggregation
        # output: [Batch, Seq, Hidden_Dim] (concatenated forward and backward states)
        x, _ = self.bigru(x)

        # Final Prediction Head
        out = self.head(x)  # [Batch, Seq, Output_Dim]

        return out
