import torch
import torch.nn as nn
from library.config import Config


class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM architecture for Ventilator Pressure Prediction.

    This model combines a 1D Convolutional Neural Network (CNN) to extract local
    dynamic features (simulating derivatives and smoothing) with a Bidirectional
    LSTM to model the sequential integration of pressure over the breath cycle.

    Architecture:
    1. Input: (Batch, Seq_Len, Input_Dim)
    2. CNN Block: Conv1d -> BatchNorm -> GELU
    3. LSTM Block: Deep Bidirectional LSTM
    4. Head: Linear projection to scalar output
    """

    def __init__(self):
        super(HybridCNNLSTM, self).__init__()

        # ==========================================
        # 1. Local Dynamics Encoder (1D CNN)
        # ==========================================
        # We use padding to preserve the sequence length (L=80)
        # Padding = (Kernel_Size - 1) // 2 (assuming odd kernel size)
        padding = (Config.CNN_KERNEL_SIZE - 1) // 2

        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=Config.INPUT_DIM,
                out_channels=Config.CNN_FILTERS,
                kernel_size=Config.CNN_KERNEL_SIZE,
                padding=padding,
                bias=False,  # Bias is redundant when using BatchNorm
            ),
            nn.BatchNorm1d(Config.CNN_FILTERS),
            nn.GELU(),
        )

        # ==========================================
        # 2. Sequential Integrator (LSTM)
        # ==========================================
        self.lstm = nn.LSTM(
            input_size=Config.CNN_FILTERS,
            hidden_size=Config.LSTM_HIDDEN_DIM,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=Config.LSTM_BIDIRECTIONAL,
            dropout=Config.LSTM_DROPOUT if Config.LSTM_LAYERS > 1 else 0.0,
        )

        # ==========================================
        # 3. Regressor Head
        # ==========================================
        # Calculate the output dimension of the LSTM
        # If bidirectional, the output size is hidden_dim * 2
        lstm_output_dim = Config.LSTM_HIDDEN_DIM * (
            2 if Config.LSTM_BIDIRECTIONAL else 1
        )

        self.head = nn.Linear(lstm_output_dim, 1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Input_Dim).

        Returns:
            torch.Tensor: Predicted pressure of shape (Batch, Seq_Len).
        """
        # Input x: (Batch, Seq_Len, Input_Dim)

        # --- CNN Block ---
        # Conv1d expects input shape (Batch, Channels, Seq_Len)
        x = x.transpose(1, 2)  # -> (Batch, Input_Dim, Seq_Len)

        x = self.cnn_encoder(x)

        # Transpose back for LSTM: (Batch, Seq_Len, CNN_Filters)
        x = x.transpose(1, 2)

        # --- LSTM Block ---
        # LSTM output: (Batch, Seq_Len, Hidden_Dim * Directions)
        # We ignore the hidden states (h_n, c_n)
        x, _ = self.lstm(x)

        # --- Regressor Head ---
        # Project to scalar pressure: (Batch, Seq_Len, 1)
        x = self.head(x)

        # Remove the last dimension to match target shape: (Batch, Seq_Len)
        return x.squeeze(-1)
