import torch
import torch.nn as nn
from library.config import Config


class ResidualDenseBlock(nn.Module):
    """
    A Residual Dense Block for the TCN branch.
    Structure: Input -> Conv -> BN -> GELU -> Dropout -> Conv -> BN -> GELU -> Dropout -> Add
    Includes a 1x1 Conv projection on the skip connection if dimensions change.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(ResidualDenseBlock, self).__init__()

        # Calculate padding to maintain sequence length (Same padding)
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        # Projection for residual connection if dimensions mismatch
        self.project = None
        if in_channels != out_channels:
            self.project = nn.Conv1d(in_channels, out_channels, kernel_size=1)

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

        if self.project is not None:
            residual = self.project(residual)

        return out + residual


class PCDRHNet(nn.Module):
    """
    Physically-Conformant Direct-Residual Hybrid Network (PCDRH-Net).

    Architecture:
    1. Direct-Temporal Interface: Raw scaled features -> Branches (No Stem).
    2. Branch 1 (Resistive): Deep Residual Dense TCN.
    3. Branch 2 (Elastic): High-Capacity Bidirectional LSTM.
    4. Fusion Head: Wide-Latent Integration -> Output.
    """

    def __init__(self):
        super(PCDRHNet, self).__init__()

        # Dynamic Input Dimension based on Stream A features
        input_dim = len(Config.STREAM_A_COLS)

        # ==========================
        # Branch 1: Deep Residual Dense TCN (Resistive Stream)
        # ==========================
        tcn_layers = []
        current_dim = input_dim

        for out_dim in Config.CNN_FILTERS:
            tcn_layers.append(
                ResidualDenseBlock(
                    in_channels=current_dim,
                    out_channels=out_dim,
                    kernel_size=Config.CNN_KERNEL_SIZE,
                    dropout=Config.CNN_DROPOUT,
                )
            )
            current_dim = out_dim

        self.tcn_branch = nn.Sequential(*tcn_layers)
        self.tcn_out_dim = Config.CNN_FILTERS[-1]

        # ==========================
        # Branch 2: High-Capacity Bidirectional LSTM (Elastic Stream)
        # ==========================
        self.lstm_branch = nn.LSTM(
            input_size=input_dim,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=Config.LSTM_BIDIRECTIONAL,
        )

        lstm_out_dim = Config.LSTM_HIDDEN_SIZE * (2 if Config.LSTM_BIDIRECTIONAL else 1)

        # ==========================
        # Fusion Head: Wide-Latent Integration
        # ==========================
        fusion_input_dim = self.tcn_out_dim + lstm_out_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, Config.DENSE_HIDDEN_SIZE),
            nn.GELU(),
            nn.Linear(Config.DENSE_HIDDEN_SIZE, 1),
        )

        # Weight Initialization
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for better convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Features).

        Returns:
            torch.Tensor: Predicted pressure of shape (Batch, Seq_Len, 1).
        """
        # x shape: [Batch, Seq, Feat]

        # ---------------------
        # Branch 1: TCN
        # ---------------------
        # Conv1d expects [Batch, Channel, Seq]
        x_tcn = x.permute(0, 2, 1)

        tcn_out = self.tcn_branch(x_tcn)

        # Permute back to [Batch, Seq, Channel] for concatenation
        tcn_out = tcn_out.permute(0, 2, 1)

        # ---------------------
        # Branch 2: LSTM
        # ---------------------
        # LSTM expects [Batch, Seq, Feat] (batch_first=True)
        lstm_out, _ = self.lstm_branch(x)

        # ---------------------
        # Fusion
        # ---------------------
        # Concatenate along the feature dimension
        combined = torch.cat([tcn_out, lstm_out], dim=2)

        # Project to output
        output = self.fusion_head(combined)

        return output
