import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import Config


class BiLSTMClassifier(nn.Module):
    """
    Bi-directional LSTM Classifier for frame-wise gesture recognition.

    Architecture:
    1. Bi-LSTM layers to capture temporal dependencies.
    2. Fully Connected (Linear) layer to map hidden states to class logits.
    """

    def __init__(self):
        super(BiLSTMClassifier, self).__init__()

        # Load hyperparameters from Config
        self.input_dim = Config.INPUT_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.num_classes = Config.NUM_CLASSES
        self.dropout = Config.DROPOUT
        self.bidirectional = Config.BIDIRECTIONAL

        # Define LSTM Layer
        # batch_first=True expects input shape (Batch, Time, Features)
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
        )

        # Define Output Layer
        # If bidirectional, the hidden state size is doubled
        self.fc_input_dim = (
            self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        )
        self.fc = nn.Linear(self.fc_input_dim, self.num_classes)

    def forward(self, x, lengths):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input features of shape (Batch, Max_Time, Features).
            lengths (torch.Tensor): Actual lengths of each sequence in the batch.

        Returns:
            torch.Tensor: Logits of shape (Batch, Max_Time, Num_Classes).
        """
        # Ensure lengths are on CPU for pack_padded_sequence
        lengths = lengths.cpu()

        # Pack the padded sequence to ignore padding during LSTM computation
        # enforce_sorted=False handles batches that might not be perfectly sorted by length
        packed_input = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # Pass through LSTM
        # packed_output contains the hidden states for each time step
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # Unpack the sequence back to padded tensor
        # output shape: (Batch, Max_Time, Hidden_Dim * Num_Directions)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Apply Linear layer to each time step
        # PyTorch Linear layer broadcasts over the time dimension automatically
        logits = self.fc(output)

        return logits
