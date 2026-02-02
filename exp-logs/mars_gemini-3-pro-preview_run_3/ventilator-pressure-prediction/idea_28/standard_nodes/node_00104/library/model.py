import torch
import torch.nn as nn
from library.config import Config


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block for the TCN branch.
    Structure: Input -> [Conv -> BN -> GELU -> Dropout] x2 -> Add Input
    """

    def __init__(self, channels, kernel_size, dropout):
        super(ResidualDenseBlock, self).__init__()

        # Calculate padding to keep dimensions the same (assuming stride=1, dilation=1)
        # padding = (kernel_size - 1) // 2 for centered padding
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, bias=False
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


class PCDRHNet(nn.Module):
    """
    Physically-Conformant Direct-Residual Hybrid Network (PCDRH-Net).

    Features:
    - Direct-Temporal Interface (No 1x1 Stem)
    - Branch 1: Deep Residual Dense TCN (Resistive Stream)
    - Branch 2: High-Capacity Bidirectional LSTM (Elastic Stream)
    - Fusion: Wide-Latent Integration
    """

    def __init__(self, input_dim):
        super(PCDRHNet, self).__init__()

        # ----------------------------------------------------------------------
        # Branch 1: Deep Residual Dense TCN (Resistive Stream)
        # ----------------------------------------------------------------------
        # Entry Layer: Direct-Temporal Interface
        # Projects raw features directly using large temporal kernels
        self.tcn_entry = nn.Conv1d(
            in_channels=input_dim,
            out_channels=Config.TCN_CHANNELS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            padding=(Config.TCN_KERNEL_SIZE - 1) // 2,
            dilation=Config.TCN_DILATION,
        )

        # Stack of Residual Dense Blocks
        self.tcn_blocks = nn.ModuleList(
            [
                ResidualDenseBlock(
                    channels=Config.TCN_CHANNELS,
                    kernel_size=Config.TCN_KERNEL_SIZE,
                    dropout=Config.TCN_DROPOUT,
                )
                for _ in range(Config.TCN_LAYERS)
            ]
        )

        # ----------------------------------------------------------------------
        # Branch 2: High-Capacity Bidirectional LSTM (Elastic Stream)
        # ----------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=Config.LSTM_BIDIRECTIONAL,
        )

        # ----------------------------------------------------------------------
        # Fusion Head: Wide-Latent Integration
        # ----------------------------------------------------------------------
        # Calculate fusion input size
        # TCN output channels + LSTM hidden size * 2 (bidirectional)
        fusion_input_dim = Config.TCN_CHANNELS + (Config.LSTM_HIDDEN_SIZE * 2)

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, Config.FUSION_HIDDEN_SIZE),
            nn.GELU(),
            nn.Linear(Config.FUSION_HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Features)
        Returns:
            out: Predicted pressure of shape (Batch, Seq_Len)
        """
        # ----------------------------------------------------------------------
        # Branch 1: TCN Forward
        # ----------------------------------------------------------------------
        # Permute for Conv1d: (Batch, Seq, Feat) -> (Batch, Feat, Seq)
        x_tcn = x.transpose(1, 2)

        # Apply entry layer (Direct-Temporal Interface)
        out_tcn = self.tcn_entry(x_tcn)

        # Apply Residual Blocks
        for block in self.tcn_blocks:
            out_tcn = block(out_tcn)

        # Permute back: (Batch, Feat, Seq) -> (Batch, Seq, Feat)
        out_tcn = out_tcn.transpose(1, 2)

        # ----------------------------------------------------------------------
        # Branch 2: LSTM Forward
        # ----------------------------------------------------------------------
        # LSTM expects (Batch, Seq, Feat)
        out_lstm, _ = self.lstm(x)

        # ----------------------------------------------------------------------
        # Fusion
        # ----------------------------------------------------------------------
        # Concatenate along feature dimension
        combined = torch.cat([out_tcn, out_lstm], dim=2)

        # Project to scalar output
        out = self.fusion_head(combined)

        # Remove last dimension: (Batch, Seq, 1) -> (Batch, Seq)
        return out.squeeze(-1)
