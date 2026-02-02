import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BiLSTM(nn.Module):
    """
    Bi-Directional LSTM for continuous gesture recognition.
    """

    def __init__(
        self,
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.LSTM_LAYERS,
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROPOUT,
    ):
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input features of shape (Batch, Time, Dim)
            mask: Optional mask (Batch, Time)
        Returns:
            logits: (Batch, Classes, Time)
        """
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        out = self.dropout(out)

        # (Batch, Time, Hidden*2) -> (Batch, Time, Classes)
        logits = self.fc(out)

        # Permute to (Batch, Classes, Time) for CrossEntropyLoss
        return logits.permute(0, 2, 1)
