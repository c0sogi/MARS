import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    INPUT_SIZE,
    NUM_CLASSES,
    LSTM_HIDDEN_DIM,
    LSTM_LAYERS,
    LSTM_DROPOUT,
    TCN_NUM_CHANNELS,
    TCN_KERNEL_SIZE,
    TCN_DROPOUT,
)


class HybridGestureNet(nn.Module):
    """
    Standard Bi-LSTM Model.
    Cite solution_lesson_node_00020: Feature Relevance Outweighs Architectural Complexity
    """

    def __init__(self):
        super(HybridGestureNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=LSTM_HIDDEN_DIM,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=LSTM_DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )
        self.fc = nn.Linear(LSTM_HIDDEN_DIM * 2, NUM_CLASSES)

    def forward(self, x, lengths=None):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Time, InputDim)
            lengths (torch.Tensor, optional): Valid lengths of sequences for packing.
        Returns:
            torch.Tensor: Logits of shape (Batch, Time, NumClasses)
        """
        if lengths is not None:
            # Move lengths to CPU for pack_padded_sequence
            lengths_cpu = lengths.cpu()
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.lstm(x_packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)

            # If the batch max length is less than x.size(1) (due to padding in collate),
            # pad_packed_sequence might return a shorter tensor. We pad it back.
            if out.size(1) < x.size(1):
                diff = x.size(1) - out.size(1)
                out = F.pad(out, (0, 0, 0, diff))
        else:
            out, _ = self.lstm(x)

        logits = self.fc(out)
        return logits
