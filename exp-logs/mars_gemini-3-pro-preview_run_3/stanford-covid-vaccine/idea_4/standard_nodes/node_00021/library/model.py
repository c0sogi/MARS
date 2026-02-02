import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class RNAGRU(nn.Module):
    """
    Bidirectional GRU-based RNA Degradation Predictor.
    Replaces Conformer to leverage inductive bias for short sequences.
    """

    def __init__(self):
        super().__init__()

        # Load hyperparameters from Config
        input_channels = Config.INPUT_CHANNELS
        hidden_dim = Config.HIDDEN_DIM
        num_layers = Config.NUM_LAYERS
        dropout = Config.DROPOUT
        output_channels = Config.OUTPUT_CHANNELS

        # 1. Input Embedding (Linear Projection)
        self.embedding = nn.Linear(input_channels, hidden_dim)

        # 2. Bidirectional GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

        # 3. Output Head
        # Input dim is hidden_dim * 2 because of bidirectionality
        self.head = nn.Linear(hidden_dim * 2, output_channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Seq_Len, 14)
        Returns:
            torch.Tensor: Predictions of shape (Batch, Seq_Len, 5)
        """
        # Project input
        x = self.embedding(x)  # (B, L, Hidden)

        # Pass through GRU
        x, _ = self.gru(x)  # (B, L, Hidden*2)

        # Project to output targets
        x = self.head(x)  # (B, L, 5)

        return x
