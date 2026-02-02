import torch
import torch.nn as nn
from library.config import Config


class SimpleConvBlock(nn.Module):
    """
    A Simplified Convolutional Block for the TCN branch.
    Ref: solution_lesson_node_00062 (Simplicity in Temporal Receptive Fields)
    Ref: solution_lesson_node_00068 (Simpler TCN blocks)
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PCSDHNet(nn.Module):
    """
    Physically-Consistent Stabilized Dense-Hybrid Network (PCSDH-Net).

    Combines:
    1. A Stabilized Deep Dense Large-Kernel TCN (Resistive Stream).
    2. A High-Capacity Bidirectional LSTM (Elastic Stream).
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------------
        # Configuration & Dimensions
        # ----------------------------------------------------------------
        input_dim = len(Config.FEATURE_COLS)
        kernel_size = Config.CNN_KERNEL_SIZE
        dropout_cnn = Config.CNN_DROPOUT
        filters = Config.CNN_FILTERS

        hidden_size = Config.LSTM_HIDDEN_SIZE
        lstm_layers = Config.LSTM_LAYERS
        bidirectional = Config.LSTM_BIDIRECTIONAL
        dropout_fc = Config.FC_DROPOUT

        # ----------------------------------------------------------------
        # Branch 1: TCN (Resistive Stream)
        # ----------------------------------------------------------------
        tcn_layers = []
        in_c = input_dim

        for out_c in filters:
            tcn_layers.append(SimpleConvBlock(in_c, out_c, kernel_size, dropout_cnn))
            in_c = out_c

        self.tcn = nn.Sequential(*tcn_layers)
        self.tcn_out_dim = filters[-1]

        # ----------------------------------------------------------------
        # Branch 2: LSTM (Elastic Stream)
        # ----------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size

        # ----------------------------------------------------------------
        # Fusion Head
        # ----------------------------------------------------------------
        fusion_dim = self.tcn_out_dim + self.lstm_out_dim

        # Ref: solution_lesson_node_00068 (Preserve capacity at fusion bottleneck)
        # Widened from 256 to 1024 to avoid information bottleneck
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout_fc),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Seq_Len, Features)
        Returns:
            torch.Tensor: Predicted pressure of shape (Batch, Seq_Len, 1)
        """
        # ----------------------------------------------------------------
        # TCN Forward Pass
        # ----------------------------------------------------------------
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x_tcn = x.transpose(1, 2)
        x_tcn = self.tcn(x_tcn)
        # Permute back: (B, C, L) -> (B, L, C)
        x_tcn = x_tcn.transpose(1, 2)

        # ----------------------------------------------------------------
        # LSTM Forward Pass
        # ----------------------------------------------------------------
        # LSTM expects (B, L, C)
        x_lstm, _ = self.lstm(x)

        # ----------------------------------------------------------------
        # Fusion
        # ----------------------------------------------------------------
        # Concatenate along feature dimension
        x_cat = torch.cat([x_tcn, x_lstm], dim=2)

        # Final Prediction
        out = self.head(x_cat)

        return out
