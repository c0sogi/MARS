import torch
import torch.nn as nn
from library.config import (
    INPUT_DIM,
    HIDDEN_DIM,
    LSTM_UNITS,
    LSTM_LAYERS,
    CNN_CHANNELS,
    KERNEL_SIZE,
    DROPOUT,
)


class ResidualDenseBlock(nn.Module):
    """
    A Residual Dense Block for the TCN branch.
    Structure: Input -> Conv -> BN -> GELU -> Dropout -> Conv -> BN -> GELU -> Dropout -> Add
    Maintains sequence length via padding and channel dimension.
    """

    def __init__(self, channels, kernel_size, dropout):
        super(ResidualDenseBlock, self).__init__()
        # Padding = (kernel_size - 1) // 2 for 'same' padding with odd kernels and stride 1
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=padding
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


class KARHNet(nn.Module):
    """
    Kinematically-Augmented Residual-Hybrid Network (KARH-Net).
    Fuses a Deep Residual Dense TCN (Resistive Stream) with a
    High-Capacity Bidirectional LSTM (Elastic Stream).
    """

    def __init__(self):
        super(KARHNet, self).__init__()

        # ==========================================
        # Branch 1: Deep Residual Dense TCN
        # ==========================================
        self.cnn_layers = nn.ModuleList()
        current_channels = CNN_CHANNELS[0]

        # Initial projection to first channel size
        self.cnn_start = nn.Conv1d(INPUT_DIM, current_channels, kernel_size=1)

        # Build blocks based on config
        # We iterate through the config. If channels change, we add a transition layer.
        # We add a ResidualDenseBlock for each stage.
        prev_channels = current_channels
        for ch in CNN_CHANNELS:
            if ch != prev_channels:
                # Transition layer to increase capacity
                self.cnn_layers.append(nn.Conv1d(prev_channels, ch, kernel_size=1))

            # Add the residual block
            self.cnn_layers.append(ResidualDenseBlock(ch, KERNEL_SIZE, DROPOUT))
            prev_channels = ch

        final_cnn_channels = CNN_CHANNELS[-1]

        # ==========================================
        # Branch 2: High-Capacity Bidirectional LSTM
        # ==========================================
        self.lstm = nn.LSTM(
            input_size=INPUT_DIM,
            hidden_size=LSTM_UNITS,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )

        # ==========================================
        # Fusion Head: Wide-Latent Integration
        # ==========================================
        # LSTM output is (Batch, Seq, Hidden*2)
        # CNN output is (Batch, Channels, Seq) -> permute to (Batch, Seq, Channels)
        fusion_input_dim = final_cnn_channels + (LSTM_UNITS * 2)

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Length, Features)
        Returns:
            Output tensor of shape (Batch, Length)
        """
        # --- LSTM Branch ---
        # x is already (Batch, Length, Features)
        lstm_out, _ = self.lstm(x)

        # --- CNN Branch ---
        # Permute to (Batch, Features, Length) for Conv1d
        cnn_x = x.permute(0, 2, 1)

        # Initial Projection
        cnn_x = self.cnn_start(cnn_x)

        # Pass through residual stack
        for layer in self.cnn_layers:
            cnn_x = layer(cnn_x)

        # Permute back to (Batch, Length, Channels)
        cnn_out = cnn_x.permute(0, 2, 1)

        # --- Fusion ---
        # Concatenate along feature dimension
        combined = torch.cat([cnn_out, lstm_out], dim=2)

        # Project
        out = self.fusion_head(combined)

        # Squeeze last dimension (Batch, Length, 1) -> (Batch, Length)
        return out.squeeze(-1)
