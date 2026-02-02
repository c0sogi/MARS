import torch
import torch.nn as nn
from library.config import Config


class DilatedResidualBiGRU(nn.Module):
    """
    Optimized BiGRU Architecture with Convolutional Stem (Cite Lesson 00026, 00027).

    Components:
    1. Convolutional Stem: Projects sparse inputs to dense embeddings and captures local context.
    2. BiGRU Backbone: High-capacity recurrent layers for global sequence modeling.
    3. Output Head: Projects hidden states to targets.
    """

    def __init__(self):
        super(DilatedResidualBiGRU, self).__init__()

        # Load hyperparameters from Config
        input_channels = Config.NUM_CHANNELS
        cnn_filters = Config.CNN_FILTERS
        kernel_size = Config.CNN_KERNEL_SIZE
        dropout_rate = Config.DROPOUT
        rnn_hidden = Config.RNN_HIDDEN_DIM
        rnn_layers = Config.RNN_LAYERS
        num_targets = Config.NUM_TARGETS

        # 1. Convolutional Stem (Cite Lesson 00027)
        # Projects (N, 14, L) -> (N, 256, L)
        # Captures local n-grams and expands dimensionality
        padding = kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels, cnn_filters, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 2. Recurrent Backbone (BiGRU) (Cite Lesson 00026)
        # Increased capacity (256 hidden dim) replacing complex encoder
        self.gru = nn.GRU(
            input_size=cnn_filters,
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
        # Permute input for CNN: (Batch, Seq_Len, Channels) -> (Batch, Channels, Seq_Len)
        x = x.permute(0, 2, 1)

        # Convolutional Stem
        x = self.stem(x)

        # Permute output for RNN: (Batch, Channels, Seq_Len) -> (Batch, Seq_Len, Channels)
        x = x.permute(0, 2, 1)

        # GRU Backbone
        x, _ = self.gru(x)

        # Output Head
        x = self.fc_dropout(x)
        out = self.fc(x)

        return out
