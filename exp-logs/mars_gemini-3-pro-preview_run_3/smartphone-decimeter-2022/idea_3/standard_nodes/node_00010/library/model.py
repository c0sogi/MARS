import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library import config


class ResidualBiLSTM(nn.Module):
    """
    A Bidirectional LSTM model that predicts position residuals (corrections)
    based on a sequence of GNSS and IMU features.

    The model outputs a correction vector (dLat, dLon) for each time step,
    which is intended to be added to a baseline WLS position.
    """

    def __init__(
        self,
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=config.OUTPUT_SIZE,
        dropout=config.DROPOUT,
        bidirectional=config.BIDIRECTIONAL,
    ):
        """
        Args:
            input_size (int): Number of input features per time step.
            hidden_size (int): Number of features in the hidden state of the LSTM.
            num_layers (int): Number of recurrent layers.
            output_size (int): Number of output values per time step (default 2 for Lat/Lon).
            dropout (float): Dropout probability.
            bidirectional (bool): If True, becomes a Bidirectional LSTM.
        """
        super(ResidualBiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Backbone: Multi-layer Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Head: Time-Distributed Linear Layer
        # If bidirectional, the hidden state dimension is doubled
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, output_size)

    def forward(self, x, lengths=None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).
            lengths (torch.Tensor, optional): Tensor of shape (batch_size) containing
                                              the valid length of each sequence.

        Returns:
            torch.Tensor: Predicted corrections of shape (batch_size, seq_len, output_size).
        """
        # Use PackedSequence for efficient processing of variable length sequences
        if lengths is not None:
            # pack_padded_sequence requires lengths to be on CPU
            lengths_cpu = lengths.cpu()

            # Create a PackedSequence object
            # enforce_sorted=False allows us to handle batches that aren't sorted by length
            packed_x = pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )

            # Pass through LSTM
            packed_out, _ = self.lstm(packed_x)

            # Unpack back to padded tensor
            # total_length ensures the output matches the input sequence length (preserving padding)
            lstm_out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=x.size(1)
            )
        else:
            # Fallback if lengths are not provided (e.g., fixed length batches without padding)
            lstm_out, _ = self.lstm(x)

        # Apply the linear head to map hidden states to coordinate corrections
        # The Linear layer in PyTorch is applied to the last dimension, effectively
        # acting as a Time-Distributed layer across the sequence.
        # Input: (batch_size, seq_len, lstm_out_dim) -> Output: (batch_size, seq_len, output_size)
        out = self.fc(lstm_out)

        return out
