import torch
import torch.nn as nn
from library.config import Config


class ResidualDenseBlock(nn.Module):
    """
    A Residual Dense Block for the TCN branch.
    Structure: Input + [Conv1D -> BN -> GELU -> Dropout -> Conv1D -> BN -> GELU -> Dropout]
    """

    def __init__(self, channels, kernel_size, dropout):
        super().__init__()
        # Calculate padding to maintain sequence length (Same padding)
        padding = (kernel_size - 1) // 2

        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Residual connection
        return x + self.block(x)


class RDHNet(nn.Module):
    """
    Residual-Dense Hybrid Network (RDH-Net).
    Combines a deep Residual Dense TCN (Resistive Stream) with a
    High-Capacity Bi-LSTM (Elastic Stream).
    """

    def __init__(self):
        super().__init__()

        input_dim = Config.get_input_dim()

        # ==========================================
        # Branch 1: Deep Residual Dense TCN (Resistive Stream)
        # ==========================================
        self.cnn_filters = Config.CNN_FILTERS
        self.kernel_size = Config.KERNEL_SIZE
        self.cnn_dropout = Config.CNN_DROPOUT

        # Stem: Project input features to CNN channel dimension
        self.cnn_stem = nn.Conv1d(input_dim, self.cnn_filters, kernel_size=1)

        # Backbone: Stack of Residual Dense Blocks
        cnn_layers = []
        for _ in range(Config.CNN_BLOCKS):
            cnn_layers.append(
                ResidualDenseBlock(
                    channels=self.cnn_filters,
                    kernel_size=self.kernel_size,
                    dropout=self.cnn_dropout,
                )
            )
        self.cnn_backbone = nn.Sequential(*cnn_layers)

        # ==========================================
        # Branch 2: High-Capacity Bi-LSTM (Elastic Stream)
        # ==========================================
        self.lstm_hidden = Config.LSTM_HIDDEN
        self.lstm_layers = Config.LSTM_LAYERS
        self.lstm_bidirectional = Config.LSTM_BIDIRECTIONAL
        self.lstm_dropout = Config.LSTM_DROPOUT

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=self.lstm_bidirectional,
            dropout=self.lstm_dropout if self.lstm_layers > 1 else 0,
        )

        # ==========================================
        # Fusion Head: Wide-Latent Integration
        # ==========================================
        # Calculate combined dimension
        # CNN output: cnn_filters
        # LSTM output: lstm_hidden * 2 (if bidirectional)
        lstm_out_dim = self.lstm_hidden * (2 if self.lstm_bidirectional else 1)
        fusion_input_dim = self.cnn_filters + lstm_out_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, Config.FUSION_HIDDEN),
            nn.GELU(),
            nn.Linear(Config.FUSION_HIDDEN, 1),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Features)
        Returns:
            torch.Tensor: Predicted pressure of shape (Batch, Seq_Len)
        """
        # x shape: (B, L, C)

        # --- Branch 1: CNN ---
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.cnn_stem(x_cnn)
        x_cnn = self.cnn_backbone(x_cnn)
        # Permute back: (B, C, L) -> (B, L, C)
        x_cnn = x_cnn.permute(0, 2, 1)

        # --- Branch 2: LSTM ---
        # LSTM expects (B, L, C)
        x_lstm, _ = self.lstm(x)

        # --- Fusion ---
        # Concatenate along feature dimension
        x_fused = torch.cat([x_cnn, x_lstm], dim=2)

        # Project to output
        out = self.fusion_head(x_fused)

        # Squeeze the last dimension: (B, L, 1) -> (B, L)
        return out.squeeze(-1)
