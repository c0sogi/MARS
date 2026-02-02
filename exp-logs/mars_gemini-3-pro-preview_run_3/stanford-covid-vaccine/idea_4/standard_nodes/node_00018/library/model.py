import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class RNAGRU(nn.Module):
    """
    Bidirectional GRU for RNA Degradation Prediction.
    Cite Lesson 00017: Prefer RNNs over Transformers for short sequences with limited data.
    """

    def __init__(self):
        super().__init__()

        # Load hyperparameters from Config
        input_channels = Config.INPUT_CHANNELS
        dim = Config.DIM_MODEL
        num_layers = Config.NUM_LAYERS
        dropout = Config.DROPOUT
        output_channels = Config.OUTPUT_CHANNELS

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output Head
        # Input dim is doubled because of bidirectionality
        self.head = nn.Linear(dim * 2, output_channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Seq_Len, 14)
        Returns:
            torch.Tensor: Predictions of shape (Batch, Seq_Len, 5)
        """
        # x: (Batch, Seq_Len, 14)

        # Pass through GRU
        x, _ = self.gru(x)  # (Batch, Seq_Len, Dim * 2)

        # Project to output targets
        x = self.head(x)  # (Batch, Seq_Len, 5)

        return x
