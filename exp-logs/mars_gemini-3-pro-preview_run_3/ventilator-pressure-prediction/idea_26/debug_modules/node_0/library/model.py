import torch
import torch.nn as nn
from library.config import Config


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block for the TCN branch.
    Structure: Input -> Conv -> BN -> GELU -> Dropout -> Conv -> BN -> GELU -> Dropout -> Add
    Uses dense convolutions (dilation=1) with large kernels to model derivatives.
    """

    def __init__(self, channels, kernel_size, dropout):
        super(ResidualDenseBlock, self).__init__()

        # First convolution block
        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="same",
            dilation=1,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        # Second convolution block
        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="same",
            dilation=1,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
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


class DKRHNet(nn.Module):
    """
    Direct-Kinematic Residual-Hybrid Network (DKRH-Net).

    Architecture:
    1. Input: Direct-Temporal Interface (Raw features -> Large Kernel Conv).
    2. Branch 1 (Resistive): Deep Residual Dense TCN.
    3. Branch 2 (Elastic): High-Capacity Bidirectional LSTM.
    4. Fusion: Concatenation -> Wide Linear Projection -> Output.
    """

    def __init__(self):
        super(DKRHNet, self).__init__()

        # Retrieve hyperparameters from Config
        input_dim = Config.INPUT_DIM

        # TCN Hyperparameters
        cnn_filters = Config.CNN_FILTERS
        cnn_kernel = Config.CNN_KERNEL_SIZE
        cnn_layers = Config.CNN_LAYERS
        cnn_dropout = Config.CNN_DROPOUT

        # LSTM Hyperparameters
        lstm_hidden = Config.LSTM_HIDDEN_SIZE
        lstm_layers = Config.LSTM_LAYERS
        lstm_bi = Config.LSTM_BIDIRECTIONAL
        lstm_dropout = Config.LSTM_DROPOUT

        # Head Hyperparameters
        dense_hidden = Config.DENSE_HIDDEN_SIZE

        # ============================================================
        # Branch 1: Resistive Stream (Deep Residual Dense TCN)
        # ============================================================
        # Entry Layer: Direct-Temporal Interface.
        # Explicitly rejects 1x1 stem. Uses large kernel to capture rates of change immediately.
        self.tcn_entry = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel,
            padding="same",
        )

        # Stack of Residual Dense Blocks
        self.tcn_blocks = nn.ModuleList(
            [
                ResidualDenseBlock(cnn_filters, cnn_kernel, cnn_dropout)
                for _ in range(cnn_layers)
            ]
        )

        # ============================================================
        # Branch 2: Elastic Stream (High-Capacity Bi-LSTM)
        # ============================================================
        # Serves as the numerical integrator.
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=lstm_bi,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
        )

        # ============================================================
        # Fusion Head: Wide-Latent Integration
        # ============================================================
        lstm_out_dim = lstm_hidden * 2 if lstm_bi else lstm_hidden
        fusion_dim = cnn_filters + lstm_out_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, dense_hidden), nn.GELU(), nn.Linear(dense_hidden, 1)
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            out: Tensor of shape (batch_size, seq_len)
        """
        # ==========================
        # Branch 1: TCN Processing
        # ==========================
        # Permute to (batch, channels, seq_len) for Conv1d
        x_tcn = x.transpose(1, 2)

        # Apply Direct-Temporal Interface
        x_tcn = self.tcn_entry(x_tcn)

        # Apply Residual Blocks
        for block in self.tcn_blocks:
            x_tcn = block(x_tcn)

        # Permute back to (batch, seq_len, channels)
        x_tcn = x_tcn.transpose(1, 2)

        # ==========================
        # Branch 2: LSTM Processing
        # ==========================
        # LSTM expects (batch, seq_len, features)
        self.lstm.flatten_parameters()  # Optimization for RNNs
        x_lstm, _ = self.lstm(x)

        # ==========================
        # Fusion & Output
        # ==========================
        # Concatenate features from both branches
        x_fused = torch.cat([x_tcn, x_lstm], dim=-1)

        # Project to output
        out = self.head(x_fused)

        # Squeeze the last dimension to match target shape (batch, seq_len)
        return out.squeeze(-1)
