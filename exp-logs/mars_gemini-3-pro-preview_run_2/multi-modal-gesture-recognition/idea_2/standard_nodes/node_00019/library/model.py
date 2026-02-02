import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BiLSTM(nn.Module):
    """
    Bi-Directional LSTM for continuous gesture recognition.
    Cite Lesson 00004: Effective for root-relative skeleton + velocity features.
    """

    def __init__(
        self,
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_RNN_LAYERS,
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROPOUT,
    ):
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, mask=None):
        """
        Args:
            x: (Batch, Time, Dim)
            mask: (Batch, Time) - Not used in basic LSTM forward but kept for API.
        Returns:
            outputs: List containing one tensor (Batch, Classes, Time) to match Trainer API.
        """
        # LSTM output: (Batch, Time, Hidden*2)
        lstm_out, _ = self.lstm(x)

        # FC output: (Batch, Time, Classes)
        logits = self.fc(lstm_out)

        # Permute to (Batch, Classes, Time) for CrossEntropyLoss
        logits = logits.permute(0, 2, 1)

        # Return as list to maintain compatibility with Trainer loop expecting MS-TCN style output list
        return [logits]
