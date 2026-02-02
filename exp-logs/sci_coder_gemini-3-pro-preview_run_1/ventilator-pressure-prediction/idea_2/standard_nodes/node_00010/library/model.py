import torch
import torch.nn as nn
from library.config import Config


class HybridCNNLSTM(nn.Module):
    """
    Physics-Enhanced Hybrid CNN-LSTM Model for Ventilator Pressure Prediction.

    Architecture:
    1. 1D Convolution: Extracts local temporal features and derivatives from control signals.
    2. Residual Bi-LSTM: Models long-term dependencies with skip connections for gradient flow.
    3. Regression Head: Projects hidden states to scalar pressure values.
    """

    def __init__(self):
        super().__init__()

        # ==========================================
        # 1. CNN Encoder
        # ==========================================
        # Input shape: (Batch, Input_Dim, Seq_Len) - requires transpose in forward
        self.cnn = nn.Conv1d(
            in_channels=Config.INPUT_DIM,
            out_channels=Config.CNN_FILTERS,
            kernel_size=Config.CNN_KERNEL_SIZE,
            padding=(Config.CNN_KERNEL_SIZE - 1) // 2,  # 'Same' padding for odd kernels
        )
        self.cnn_activation = nn.GELU()

        # ==========================================
        # 2. Residual Bidirectional LSTM Stack
        # ==========================================
        self.lstm_layers = nn.ModuleList()
        self.lstm_dropouts = nn.ModuleList()

        # Dimensions
        self.bidirectional = Config.BIDIRECTIONAL
        num_directions = 2 if self.bidirectional else 1

        lstm_input_dim = Config.CNN_FILTERS
        lstm_hidden_dim = Config.LSTM_HIDDEN_SIZE
        lstm_output_dim = lstm_hidden_dim * num_directions

        for i in range(Config.LSTM_LAYERS):
            # The first layer maps from CNN filters to LSTM hidden space.
            # Subsequent layers map from LSTM output to LSTM output.
            input_size = lstm_input_dim if i == 0 else lstm_output_dim

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=lstm_hidden_dim,
                    batch_first=True,
                    bidirectional=self.bidirectional,
                )
            )

            self.lstm_dropouts.append(nn.Dropout(Config.LSTM_DROPOUT))

        # ==========================================
        # 3. Regression Head
        # ==========================================
        self.head = nn.Sequential(
            nn.Linear(lstm_output_dim, Config.FC_HIDDEN_SIZE),
            nn.GELU(),
            nn.Linear(Config.FC_HIDDEN_SIZE, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Input_Dim).

        Returns:
            torch.Tensor: Predicted pressure of shape (Batch, Seq_Len).
        """
        # 1. CNN Encoder
        # Permute to (Batch, Input_Dim, Seq_Len) for Conv1d
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = self.cnn_activation(x)

        # Permute back to (Batch, Seq_Len, CNN_Filters) for LSTM
        x = x.transpose(1, 2)

        # 2. LSTM Stack with Residuals
        for lstm, dropout in zip(self.lstm_layers, self.lstm_dropouts):
            # Save identity for residual connection
            identity = x

            # Forward through LSTM layer
            # LSTM returns (output, (h_n, c_n)), we only need output
            out, _ = lstm(x)
            out = dropout(out)

            # Apply residual connection if shapes match
            # (Layer 0 usually changes dimension, so no residual there)
            if out.shape == identity.shape:
                x = out + identity
            else:
                x = out

        # 3. Regression Head
        # Input: (Batch, Seq_Len, Hidden*Directions)
        # Output: (Batch, Seq_Len, 1)
        x = self.head(x)

        # Squeeze the last dimension to match target shape (Batch, Seq_Len)
        return x.squeeze(-1)
