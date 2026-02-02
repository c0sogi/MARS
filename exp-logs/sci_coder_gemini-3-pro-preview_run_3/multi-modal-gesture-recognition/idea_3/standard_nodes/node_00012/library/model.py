import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    INPUT_DIM,
    GRU_HIDDEN_DIM,
    GRU_NUM_LAYERS,
    DROPOUT,
    NUM_CLASSES,
    TCN_NUM_CHANNELS,
    TCN_KERNEL_SIZE,
    TCN_DROPOUT,
)


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Sequence Encoder using Bi-directional GRU.
    Extracts local temporal dynamics and generates initial frame-wise class logits.
    """

    def __init__(self):
        super(BiGRUEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=INPUT_DIM,
            hidden_size=GRU_HIDDEN_DIM,
            num_layers=GRU_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if GRU_NUM_LAYERS > 1 else 0.0,
        )

        # Output layer: Maps from (Hidden * 2) to NumClasses
        self.fc = nn.Linear(GRU_HIDDEN_DIM * 2, NUM_CLASSES)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # x shape: (Batch, Time, InputDim)

        # GRU Output: (Batch, Time, Hidden * 2)
        out, _ = self.gru(x)

        out = self.dropout(out)

        # Project to classes: (Batch, Time, NumClasses)
        logits = self.fc(out)

        return logits
