import torch
import torch.nn as nn
from library import config


class ResidualDenseBlock(nn.Module):
    """
    A Residual Dense Block for the TCN branch.
    Structure: Input -> Conv -> BN -> GELU -> Conv -> BN -> GELU -> Dropout -> Add
    Uses large kernels to smooth physical signals and dense connections (dilation=1).
    """

    def __init__(self, channels, kernel_size, dropout):
        super(ResidualDenseBlock, self).__init__()
        # Padding is set to maintain sequence length (centered padding)
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.GELU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.dropout(out)
        return residual + out


class CWDHNet(nn.Module):
    """
    Corrected Wide-Context Dense-Hybrid Network (CWDH-Net).

    Optimized based on lessons:
    - Removed Positional Encodings (Cite solution_lesson_node_00085)
    - Increased CNN Capacity (Cite solution_lesson_node_00057)
    - Uses Residual Dense Blocks (Cite solution_lesson_node_00075)

    Features:
    - Branch 1: Deep Residual Dense TCN (Resistive Stream)
    - Branch 2: High-Capacity Bi-LSTM (Elastic Stream)
    - Fusion: Wide-Latent Integration Head
    """

    def __init__(self):
        super(CWDHNet, self).__init__()

        # =====================================================================
        # Branch 1: Deep Residual Dense TCN (Resistive Stream)
        # =====================================================================
        # Stem convolution to project input features to CNN filter dimension
        self.cnn_stem = nn.Conv1d(config.INPUT_DIM, config.CNN_FILTERS, kernel_size=1)

        # Deep stack of Residual Dense Blocks
        # We use 4 blocks to form a "Deep" stack as per the architectural idea
        self.cnn_blocks = nn.ModuleList(
            [
                ResidualDenseBlock(
                    channels=config.CNN_FILTERS,
                    kernel_size=config.CNN_KERNEL_SIZE,
                    dropout=config.CNN_DROPOUT,
                )
                for _ in range(4)
            ]
        )

        # =====================================================================
        # Branch 2: High-Capacity Bi-LSTM (Elastic Stream)
        # =====================================================================
        self.lstm = nn.LSTM(
            input_size=config.INPUT_DIM,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=config.LSTM_BIDIRECTIONAL,
        )

        # =====================================================================
        # Fusion Head: Wide-Latent Integration
        # =====================================================================
        # Calculate fusion dimension
        lstm_out_dim = (
            config.LSTM_HIDDEN_SIZE * 2
            if config.LSTM_BIDIRECTIONAL
            else config.LSTM_HIDDEN_SIZE
        )
        fusion_input_dim = config.CNN_FILTERS + lstm_out_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, config.WIDE_HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(config.FINAL_DROPOUT),
            nn.Linear(config.WIDE_HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Features)
        Returns:
            out: Prediction tensor of shape (Batch, Seq_Len)
        """
        # ---------------------------------------------------------------------
        # Branch 1: CNN Processing
        # ---------------------------------------------------------------------
        # Permute to (Batch, Features, Seq_Len) for Conv1d
        x_cnn = x.permute(0, 2, 1)

        x_cnn = self.cnn_stem(x_cnn)
        for block in self.cnn_blocks:
            x_cnn = block(x_cnn)

        # Permute back to (Batch, Seq_Len, Features) for fusion
        x_cnn = x_cnn.permute(0, 2, 1)

        # ---------------------------------------------------------------------
        # Branch 2: LSTM Processing
        # ---------------------------------------------------------------------
        # LSTM expects (Batch, Seq_Len, Features)
        x_lstm, _ = self.lstm(x)

        # ---------------------------------------------------------------------
        # Fusion and Output
        # ---------------------------------------------------------------------
        # Concatenate along the feature dimension (dim=2)
        x_fused = torch.cat([x_cnn, x_lstm], dim=2)

        # Project through wide latent layer
        out = self.fusion_head(x_fused)

        # Squeeze the last dimension to return (Batch, Seq_Len)
        return out.squeeze(-1)
