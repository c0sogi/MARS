import torch
import torch.nn as nn
from library.config import Config


class ConvBiGRU(nn.Module):
    """
    Standard BiGRU with a Convolutional Stem.

    Architecture:
    1. Input (N, 107, 14)
    2. Conv1d Stem -> Projects to dense features
    3. BiGRU -> Captures sequential context
    4. Linear Head -> Predicts 5 targets
    """

    def __init__(self):
        super(ConvBiGRU, self).__init__()

        # 1. Convolutional Stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=Config.INPUT_DIM,
                out_channels=Config.STEM_FILTERS,
                kernel_size=Config.STEM_KERNEL_SIZE,
                padding=Config.STEM_KERNEL_SIZE // 2,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 2. BiGRU Backbone
        self.gru = nn.GRU(
            input_size=Config.STEM_FILTERS,
            hidden_size=Config.RNN_HIDDEN_DIM,
            num_layers=Config.RNN_LAYERS,
            bidirectional=True,
            batch_first=True,
            dropout=Config.RNN_DROPOUT if Config.RNN_LAYERS > 1 else 0,
        )

        # 3. Output Head
        self.head = nn.Linear(Config.RNN_HIDDEN_DIM * 2, Config.OUTPUT_DIM)

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Dim)

        # Stem
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = x.transpose(1, 2)

        # GRU
        x, _ = self.gru(x)

        # Head
        out = self.head(x)

        return out
