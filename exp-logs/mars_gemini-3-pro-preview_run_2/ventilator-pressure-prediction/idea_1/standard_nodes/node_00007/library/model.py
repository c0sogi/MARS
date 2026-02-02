import torch
import torch.nn as nn
from library.config import Config


class BiLSTMRegressor(nn.Module):
    """
    Stacked Bidirectional LSTM Network for Ventilator Pressure Prediction.

    This model processes a sequence of time steps representing a breath and predicts
    the airway pressure for each time step.
    """

    def __init__(self):
        super(BiLSTMRegressor, self).__init__()

        # Calculate the dimension of the LSTM output
        # If bidirectional, the output dimension is hidden_dim * 2
        self.lstm_output_dim = (
            Config.HIDDEN_DIM * 2 if Config.BIDIRECTIONAL else Config.HIDDEN_DIM
        )

        # Recurrent Encoder: Stacked Bidirectional LSTM
        # Captures temporal dependencies in both forward and backward directions
        self.lstm = nn.LSTM(
            input_size=Config.INPUT_DIM,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.DROPOUT if Config.NUM_LAYERS > 1 else 0.0,
        )

        # Projection Head: Fully Connected Layer
        # Maps the high-dimensional hidden state to a single scalar pressure value
        self.head = nn.Linear(in_features=self.lstm_output_dim, out_features=1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Input_Dim).

        Returns:
            torch.Tensor: Predicted pressures of shape (Batch, Seq_Len).
        """
        # Pass input through the LSTM layers
        # self.lstm returns: output, (h_n, c_n)
        # output shape: (Batch, Seq_Len, lstm_output_dim)
        lstm_out, _ = self.lstm(x)

        # Project the LSTM output to the target variable (pressure)
        # linear_out shape: (Batch, Seq_Len, 1)
        linear_out = self.head(lstm_out)

        # Remove the last dimension to match the target shape (Batch, Seq_Len)
        predictions = linear_out.squeeze(-1)

        return predictions
