import torch
import torch.nn as nn
from library import config


class DilatedResidualBlock(nn.Module):
    """
    Single-layer dilated residual block.
    Structure: Conv1d -> ReLU -> Dropout -> Residual Connection.
    """

    def __init__(self, channels, kernel_size, dilation, dropout_rate):
        super().__init__()
        # Calculate padding to maintain sequence length: p = dilation * (k - 1) / 2
        # This assumes kernel_size is odd (e.g., 3).
        padding = dilation * (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.activation(out)
        out = self.dropout(out)
        return residual + out


class PartnerAwareHybridNet(nn.Module):
    """
    Partner-Aware Multi-Scale Hybrid Network.

    Features:
    - Partner-augmented input representation (handled by data loader, processed here).
    - Dilated TCN backbone with exponential dilations for global receptive field.
    - Multi-scale fusion of features from all dilation levels.
    - BiGRU for global sequence aggregation.
    """

    def __init__(self):
        super().__init__()

        # Configuration
        in_channels = config.INPUT_CHANNELS
        hidden_channels = config.CNN_CHANNELS
        kernel_size = config.KERNEL_SIZE
        dilations = config.DILATIONS
        dropout_rate = config.DROPOUT
        rnn_hidden = config.RNN_HIDDEN_SIZE
        rnn_layers = config.RNN_LAYERS
        bidirectional = config.RNN_BIDIRECTIONAL
        num_targets = len(config.TARGET_COLS)

        # 1. Stem: Project input features (18) to hidden dimension (128)
        self.stem = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        # 2. Backbone: Stack of Dilated Residual Blocks
        self.blocks = nn.ModuleList()
        for dilation in dilations:
            block = DilatedResidualBlock(
                channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout_rate=dropout_rate,
            )
            self.blocks.append(block)

        # 3. Multi-Scale Fusion
        # Concatenate outputs from all blocks and project back to hidden_channels.
        # This fuses local (low dilation) and global (high dilation) features.
        total_concat_channels = hidden_channels * len(dilations)
        self.fusion_projection = nn.Conv1d(
            total_concat_channels, hidden_channels, kernel_size=1
        )

        # 4. Global Aggregation: BiGRU
        # Hidden size is set to half of CNN channels so bidirectional output matches CNN output dim.
        self.gru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # 5. Output Head
        # If bidirectional, output dim is hidden_size * 2
        gru_out_dim = rnn_hidden * 2 if bidirectional else rnn_hidden
        self.head = nn.Linear(gru_out_dim, num_targets)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Channels) -> (B, 107, 18)
        Returns:
            logits: Output tensor of shape (Batch, Seq_Len, Targets) -> (B, 107, 5)
        """
        # Permute to (Batch, Channels, Seq_Len) for Conv1d
        x = x.permute(0, 2, 1)

        # Stem
        x = self.stem(x)

        # Backbone with Multi-Scale Feature Collection
        block_outputs = []
        current_features = x

        for block in self.blocks:
            current_features = block(current_features)
            block_outputs.append(current_features)

        # Fusion: Concatenate all scales along the channel dimension
        fused_features = torch.cat(block_outputs, dim=1)

        # Project back to standard hidden dimension
        fused_features = self.fusion_projection(fused_features)

        # Permute back to (Batch, Seq_Len, Channels) for RNN
        fused_features = fused_features.permute(0, 2, 1)

        # GRU Aggregation
        # gru_out shape: (Batch, Seq_Len, Num_Directions * Hidden_Size)
        gru_out, _ = self.gru(fused_features)

        # Prediction Head
        logits = self.head(gru_out)

        return logits
