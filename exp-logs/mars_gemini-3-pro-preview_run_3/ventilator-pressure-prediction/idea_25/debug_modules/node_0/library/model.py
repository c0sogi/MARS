import torch
import torch.nn as nn
from library.config import Config


class ResidualDenseBlock(nn.Module):
    """
    A Residual Dense Block for the TCN branch.
    Structure: Input -> Conv -> BN -> GELU -> Dropout -> Conv -> BN -> GELU -> Dropout -> Add

    Implements large-kernel dense convolutions for high-fidelity local derivative modeling.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(ResidualDenseBlock, self).__init__()

        # Calculate padding to maintain sequence length (Same/Centered padding)
        # For k=9, padding=4. Output L = L + 2*4 - (9-1) = L
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        # Shortcut connection
        # If dimensions change, use a 1x1 convolution to match them
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.drop2(out)

        return out + residual


class MCRHNet(nn.Module):
    """
    Momentum-Corrected Residual-Hybrid Network (MCRH-Net).

    Architecture:
    1. Branch 1 (Resistive Stream): Deep Residual Dense TCN
       - Models high-frequency, derivative-dependent dynamics (P ~ R*dV).
       - Uses large kernels (9) and dense convolutions (dilation=1).

    2. Branch 2 (Elastic Stream): High-Capacity Bi-LSTM
       - Models low-frequency, integral-dependent dynamics (P ~ V/C).
       - 3 layers, 512 hidden units.

    3. Fusion Head:
       - Concatenates branches.
       - Wide latent projection (1024 units).
       - No input-output skip connections to force integration.
    """

    def __init__(self, input_dim):
        super(MCRHNet, self).__init__()

        # ==========================================
        # Branch 1: Deep Residual Dense TCN
        # ==========================================

        # Stem: Project input features to the initial channel dimension
        self.tcn_stem = nn.Conv1d(input_dim, Config.CNN_CHANNELS[0], kernel_size=1)

        self.tcn_blocks = nn.ModuleList()
        in_c = Config.CNN_CHANNELS[0]

        # First block: Process at initial resolution (e.g., 64 -> 64)
        self.tcn_blocks.append(
            ResidualDenseBlock(in_c, in_c, Config.CNN_KERNEL_SIZE, Config.CNN_DROPOUT)
        )

        # Subsequent blocks: Progressive channel increase (e.g., 64->128->256->512)
        for out_c in Config.CNN_CHANNELS[1:]:
            self.tcn_blocks.append(
                ResidualDenseBlock(
                    in_c, out_c, Config.CNN_KERNEL_SIZE, Config.CNN_DROPOUT
                )
            )
            in_c = out_c

        tcn_out_dim = Config.CNN_CHANNELS[-1]

        # ==========================================
        # Branch 2: High-Capacity Bi-LSTM
        # ==========================================
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.LSTM_DROPOUT if Config.LSTM_LAYERS > 1 else 0,
        )

        # Bi-LSTM outputs 2 * hidden_size features
        lstm_out_dim = Config.LSTM_HIDDEN_SIZE * 2

        # ==========================================
        # Fusion Head
        # ==========================================
        fusion_in_dim = tcn_out_dim + lstm_out_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, Config.FUSION_HIDDEN_SIZE),
            nn.GELU(),
            nn.Linear(Config.FUSION_HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Seq_Len, Features)

        Returns:
            torch.Tensor: Predicted pressure of shape (Batch, Seq_Len, 1)
        """
        # ------------------------------------------
        # TCN Branch Forward
        # ------------------------------------------
        # Conv1d expects (Batch, Channels, Seq_Len)
        x_tcn = x.transpose(1, 2)

        x_tcn = self.tcn_stem(x_tcn)

        for block in self.tcn_blocks:
            x_tcn = block(x_tcn)

        # Transpose back to (Batch, Seq_Len, Channels)
        x_tcn = x_tcn.transpose(1, 2)

        # ------------------------------------------
        # LSTM Branch Forward
        # ------------------------------------------
        # LSTM expects (Batch, Seq_Len, Features)
        x_lstm, _ = self.lstm(x)

        # ------------------------------------------
        # Fusion
        # ------------------------------------------
        # Concatenate along feature dimension
        x_cat = torch.cat([x_tcn, x_lstm], dim=-1)

        # Project to output
        out = self.fusion(x_cat)

        return out
