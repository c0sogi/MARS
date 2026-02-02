import torch
import torch.nn as nn
from library.config import Config


class DilatedResidualBiGRU(nn.Module):
    """
    High-Capacity BiGRU Architecture.
    Simplified based on Lesson 00026: Prioritize Backbone Capacity Over Hierarchical Feature Extraction.

    Components:
    1. Local Feature Projection: Conv1d to mix local channels.
    2. BiGRU Backbone: High capacity (256 hidden dim) to capture global dependencies.
    3. Output Head.
    """

    def __init__(self):
        super(DilatedResidualBiGRU, self).__init__()

        # Load hyperparameters from Config
        input_channels = Config.NUM_CHANNELS
        # Using CNN_FILTERS as the projection dimension (matched to RNN hidden in logic)
        proj_dim = Config.RNN_HIDDEN_DIM
        kernel_size = 3
        dropout_rate = Config.DROPOUT
        rnn_hidden = Config.RNN_HIDDEN_DIM
        rnn_layers = Config.RNN_LAYERS
        num_targets = Config.NUM_TARGETS

        # 1. Feature Projection (Local context mixing)
        # Input: (N, 14, L) -> Output: (N, 256, L)
        self.projection = nn.Sequential(
            nn.Conv1d(
                input_channels,
                proj_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 2. Recurrent Backbone (BiGRU)
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if rnn_layers > 1 else 0,
        )

        # 3. Output Head
        self.fc_dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_hidden * 2, num_targets)

    def forward(self, x):
        # x: (Batch, Seq_Len, Channels)

        # Permute for Conv1d: (Batch, Channels, Seq_Len)
        x = x.permute(0, 2, 1)

        # Projection
        x = self.projection(x)

        # Permute for GRU: (Batch, Seq_Len, Channels)
        x = x.permute(0, 2, 1)

        # GRU Backbone
        x, _ = self.gru(x)

        # Output Head
        x = self.fc_dropout(x)
        out = self.fc(x)

        return out
