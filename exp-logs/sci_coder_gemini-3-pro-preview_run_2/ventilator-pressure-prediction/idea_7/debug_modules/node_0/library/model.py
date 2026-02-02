import torch
import torch.nn as nn
from library.config import (
    INPUT_DIM,
    PROJECTION_DIM,
    HIDDEN_DIM,
    NUM_LSTM_LAYERS,
    DROPOUT,
    USE_LAYER_NORM,
)


class DPI_BiLSTM(nn.Module):
    """
    Deep Projected-Injection BiLSTM (DPI-BiLSTM)

    A unified deep recurrent architecture designed to solve the 'Signal Dilution' problem.
    It projects raw features into a high-dimensional latent space and injects this
    context signal into every recurrent layer, allowing deep layers to maintain
    direct access to physical control dynamics.
    """

    def __init__(
        self,
        input_dim=INPUT_DIM,
        projection_dim=PROJECTION_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LSTM_LAYERS,
        dropout=DROPOUT,
        use_layer_norm=USE_LAYER_NORM,
    ):
        super(DPI_BiLSTM, self).__init__()

        self.use_layer_norm = use_layer_norm

        # 1. Latent Input Projection (Structural Innovation)
        # Projects ~15 raw features to a high-dimensional context (e.g., 512 dim)
        # Creates a rich "Context Signal" for the differential equations
        self.projection = nn.Sequential(nn.Linear(input_dim, projection_dim), nn.GELU())

        # 2. Deep Recurrent Backbone with Projected Injection
        self.lstm_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            # Calculate input size for each layer
            # Layer 0: Input is just the Projected Signal
            # Layer >0: Input is (Previous Layer Output) + (Projected Signal Injection)
            if i == 0:
                layer_input_dim = projection_dim
            else:
                # Previous output is hidden_dim * 2 (Bidirectional)
                layer_input_dim = (hidden_dim * 2) + projection_dim

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Stabilization via Layer Normalization
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_dim * 2))

        self.dropout = nn.Dropout(dropout)

        # 3. Output Head
        # Maps the final hidden state to the target pressure
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Input_Dim)

        Returns:
            torch.Tensor: Predicted pressure of shape (Batch, Seq_Len, 1)
        """
        # Generate the high-dimensional Context Signal
        # Shape: (Batch, Seq_Len, Projection_Dim)
        projected_x = self.projection(x)

        # Initialize the input for the first recurrent layer
        current_input = projected_x

        for i, lstm in enumerate(self.lstm_layers):
            # Latent Input Injection Logic:
            # For layers deeper than the first, we concatenate the original
            # Context Signal (projected_x) to the output of the previous layer.
            # This ensures deep layers see the physical controls directly.
            if i > 0:
                current_input = torch.cat([current_input, projected_x], dim=-1)

            # Pass through BiLSTM layer
            # lstm_out shape: (Batch, Seq_Len, Hidden_Dim * 2)
            lstm_out, _ = lstm(current_input)

            # Apply Stabilization and Regularization
            if self.use_layer_norm:
                lstm_out = self.norms[i](lstm_out)

            lstm_out = self.dropout(lstm_out)

            # The output of this layer becomes the "state" component
            # of the input for the next layer
            current_input = lstm_out

        # Final Projection to Target
        pressure = self.head(current_input)

        return pressure
