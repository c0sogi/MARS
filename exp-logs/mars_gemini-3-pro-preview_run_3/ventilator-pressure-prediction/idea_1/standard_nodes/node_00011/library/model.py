import torch
import torch.nn as nn
from library.config import Config


class PhysicsLSTM(nn.Module):
    """
    Physics-Augmented LSTM model for ventilator pressure prediction.

    Upgraded from GRU to LSTM for better handling of long-term dependencies in deeper networks.
    Cite solution_lesson_node_00001: Using physics-based features with RNNs.

    Architecture:
        Input -> Bidirectional LSTM (4 layers) -> Linear -> Output
    """

    def __init__(
        self,
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        bidirectional=Config.BIDIRECTIONAL,
        dropout=Config.DROPOUT,
    ):
        super(PhysicsLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Recurrent Core
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output Head
        head_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.head = nn.Linear(head_input_dim, 1)

    def forward(self, x):
        # Pass through LSTM
        # lstm_out shape: (batch_size, seq_len, num_directions * hidden_dim)
        lstm_out, _ = self.lstm(x)

        # Project to scalar pressure value
        predictions = self.head(lstm_out)

        return predictions.squeeze(-1)
