import torch
import torch.nn as nn
from library.config import Config


class PhysicsGRU(nn.Module):
    """
    Physics-Augmented Shallow GRU model for ventilator pressure prediction.

    This model implements a lightweight recurrent architecture designed to leverage
    physics-based features. It consists of a bidirectional GRU layer to capture
    temporal dynamics followed by a linear projection head.

    Architecture:
        Input -> Bidirectional GRU -> Linear -> Output
    """

    def __init__(
        self,
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        bidirectional=Config.BIDIRECTIONAL,
        dropout=Config.DROPOUT,
    ):
        """
        Initialize the PhysicsGRU model.

        Args:
            input_dim (int): Number of input features per time step.
            hidden_dim (int): Number of features in the hidden state of the GRU.
            num_layers (int): Number of recurrent layers.
            bidirectional (bool): If True, becomes a bidirectional GRU.
            dropout (float): Dropout probability for the GRU (only if num_layers > 1).
        """
        super(PhysicsGRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Recurrent Core
        # batch_first=True expects input shape (batch_size, seq_len, input_dim)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output Head
        # If bidirectional, the output features from GRU are doubled (forward + backward)
        head_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.head = nn.Linear(head_input_dim, 1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Predicted pressure of shape (batch_size, seq_len).
        """
        # Pass through GRU
        # gru_out shape: (batch_size, seq_len, num_directions * hidden_dim)
        # _ (hidden state) is ignored as we use the output at every time step
        gru_out, _ = self.gru(x)

        # Project to scalar pressure value
        # predictions shape: (batch_size, seq_len, 1)
        predictions = self.head(gru_out)

        # Squeeze the last dimension to match target shape (batch_size, seq_len)
        return predictions.squeeze(-1)
