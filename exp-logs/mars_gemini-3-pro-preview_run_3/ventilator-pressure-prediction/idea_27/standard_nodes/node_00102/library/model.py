import torch
import torch.nn as nn
from library.config import Config


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block for the TCN branch.
    Structure: Input -> Conv -> BN -> GELU -> Dropout -> Conv -> BN -> GELU -> Dropout -> Add
    Designed to model high-frequency, derivative-dependent dynamics.
    """

    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        # First convolution sequence
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding="same",
            dilation=1,  # Strictly dense
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        # Second convolution sequence
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding="same",
            dilation=1,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.drop2(out)

        return out + residual


class LGRHNet(nn.Module):
    """
    Logic-Gated Residual-Hybrid Network (LGRH-Net).

    Architecture:
    1. Direct-Temporal Interface (No 1x1 Stem)
    2. Branch 1: Deep Residual Dense TCN (Resistive Stream)
    3. Branch 2: High-Capacity Bidirectional LSTM (Elastic Stream)
    4. Fusion Head: Wide-Latent Integration
    """

    def __init__(self, input_dim: int, config: Config):
        super().__init__()

        # ==========================================
        # Branch 1: Deep Residual Dense TCN
        # ==========================================
        # Direct-Temporal Interface:
        # Raw features fed directly into large kernel convolution to act as
        # learnable finite-difference operators.
        self.tcn_entry = nn.Conv1d(
            input_dim,
            config.tcn_filters,
            kernel_size=config.tcn_kernel_size,
            padding="same",
        )

        # Stack of Residual Dense Blocks
        self.tcn_blocks = nn.ModuleList(
            [
                ResidualDenseBlock(
                    channels=config.tcn_filters,
                    kernel_size=config.tcn_kernel_size,
                    dropout=config.tcn_dropout,
                )
                for _ in range(config.tcn_layers)
            ]
        )

        # ==========================================
        # Branch 2: High-Capacity Bi-LSTM
        # ==========================================
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=config.lstm_bidirectional,
            dropout=config.lstm_dropout if config.lstm_layers > 1 else 0,
        )

        # ==========================================
        # Fusion Head
        # ==========================================
        lstm_out_dim = config.lstm_hidden_size * (2 if config.lstm_bidirectional else 1)
        fusion_in_dim = config.tcn_filters + lstm_out_dim

        # Wide-Latent Integration
        self.head = nn.Sequential(
            nn.Linear(fusion_in_dim, config.fusion_hidden_size),
            nn.GELU(),
            nn.Linear(config.fusion_hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Seq_Len, Features)
        Returns:
            torch.Tensor: Predicted pressure of shape (Batch, Seq_Len, 1)
        """
        # ==========================================
        # TCN Branch Execution
        # ==========================================
        # Transpose to (Batch, Features, Seq_Len) for Conv1d
        x_tcn = x.transpose(1, 2)

        # Entry (Direct Interface)
        x_tcn = self.tcn_entry(x_tcn)

        # Residual Blocks
        for block in self.tcn_blocks:
            x_tcn = block(x_tcn)

        # Transpose back to (Batch, Seq_Len, Filters)
        x_tcn = x_tcn.transpose(1, 2)

        # ==========================================
        # LSTM Branch Execution
        # ==========================================
        # LSTM expects (Batch, Seq_Len, Features)
        x_lstm, _ = self.lstm(x)

        # ==========================================
        # Fusion & Prediction
        # ==========================================
        # Concatenate along feature dimension
        x_fused = torch.cat([x_tcn, x_lstm], dim=2)

        # Project to output
        out = self.head(x_fused)

        return out
