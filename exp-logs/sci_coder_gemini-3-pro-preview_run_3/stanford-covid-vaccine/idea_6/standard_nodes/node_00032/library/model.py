import torch
import torch.nn as nn
from library.config import Config


class ResidualBiGRU(nn.Module):
    """
    A Bidirectional GRU layer with a residual connection.
    If the input dimension differs from the output dimension (2 * hidden_dim),
    a linear projection is applied to the residual path.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super(ResidualBiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * 2

        # Bidirectional GRU
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        # Projection for residual connection if dimensions don't match
        if input_dim != self.output_dim:
            self.residual_proj = nn.Linear(input_dim, self.output_dim)
        else:
            self.residual_proj = nn.Identity()

        # Optional: LayerNorm could be added here for stability,
        # but we stick to the prompt's description of Residual Connections.
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Dim)

        # GRU Forward
        # out: (Batch, Seq_Len, Hidden_Dim * 2)
        out, _ = self.gru(x)
        out = self.dropout(out)

        # Residual Connection
        res = self.residual_proj(x)

        # Add and Normalize
        return self.layer_norm(res + out)


class DeepResBiGRU(nn.Module):
    """
    Deep Residual BiGRU with a Convolutional Stem.

    Architecture:
    1. Input (N, 107, 14)
    2. Conv1d Stem -> Projects to dense features
    3. Stack of Residual BiGRU layers -> Captures sequential context
    4. Linear Head -> Predicts 5 targets
    """

    def __init__(self):
        super(DeepResBiGRU, self).__init__()

        # ------------------------------------------------------------------
        # 1. Convolutional Stem
        # ------------------------------------------------------------------
        # Input: (N, 14, 107) after permute
        # Output: (N, STEM_FILTERS, 107)
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=Config.INPUT_DIM,
                out_channels=Config.STEM_FILTERS,
                kernel_size=Config.STEM_KERNEL_SIZE,
                padding=Config.STEM_KERNEL_SIZE // 2,  # Maintain sequence length
            ),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ------------------------------------------------------------------
        # 2. Deep Residual BiGRU Backbone
        # ------------------------------------------------------------------
        self.rnn_layers = nn.ModuleList()

        # First layer input dimension is the stem output dimension
        input_dim = Config.STEM_FILTERS

        for _ in range(Config.RNN_LAYERS):
            self.rnn_layers.append(
                ResidualBiGRU(
                    input_dim=input_dim,
                    hidden_dim=Config.RNN_HIDDEN_DIM,
                    dropout=Config.RNN_DROPOUT,
                )
            )
            # Subsequent layers take the output of the previous BiGRU (2 * hidden)
            input_dim = Config.RNN_HIDDEN_DIM * 2

        # ------------------------------------------------------------------
        # 3. Output Head
        # ------------------------------------------------------------------
        self.head = nn.Linear(input_dim, Config.OUTPUT_DIM)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Input_Dim)
                              e.g., (N, 107, 14)
        Returns:
            torch.Tensor: Output tensor of shape (Batch, Seq_Len, Output_Dim)
                          e.g., (N, 107, 5)
        """
        # 1. Stem (Conv1d expects Channel first)
        # x: (N, L, C) -> (N, C, L)
        x = x.transpose(1, 2)
        x = self.stem(x)
        # x: (N, C, L) -> (N, L, C)
        x = x.transpose(1, 2)

        # 2. Residual BiGRU Backbone
        for layer in self.rnn_layers:
            x = layer(x)

        # 3. Head
        out = self.head(x)

        return out
