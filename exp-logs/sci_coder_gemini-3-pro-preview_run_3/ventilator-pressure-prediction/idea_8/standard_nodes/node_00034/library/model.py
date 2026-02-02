import torch
import torch.nn as nn
from library.config import Config


class PSANet(nn.Module):
    """
    Physically-Structured Additive Hybrid Network (PSA-Net).

    This architecture explicitly models the governing equation of motion for the lung
    (P = P_resistive + P_elastic + P_nonlinear) by decomposing the network into
    specialized parallel branches that sum up to the final prediction.

    Structure:
    1. Resistive Branch: Pyramidal Wide-Kernel TCN (models R * Flow dynamics).
    2. Elastic Branch: Deep Bidirectional LSTM (models Volume / C dynamics).
    3. Fusion Head: Additive Residual Decomposition.
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # ----------------------------------------------------------------------
        # Branch 1: Pyramidal Wide-Kernel TCN (The Resistive Stream)
        # ----------------------------------------------------------------------
        # Designed to capture fast, derivative-based signal changes.
        tcn_layers = []
        in_channels = config.INPUT_DIM
        kernel_size = config.TCN_KERNEL_SIZE
        # Calculate 'same' padding: (k - 1) // 2
        padding = (kernel_size - 1) // 2

        for out_channels in config.TCN_CHANNELS:
            tcn_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
            )
            tcn_layers.append(nn.BatchNorm1d(out_channels))
            tcn_layers.append(nn.GELU())
            tcn_layers.append(nn.Dropout(config.DROPOUT))
            in_channels = out_channels

        self.tcn_branch = nn.Sequential(*tcn_layers)
        self.tcn_out_dim = config.TCN_CHANNELS[-1]

        # ----------------------------------------------------------------------
        # Branch 2: High-Capacity Bidirectional LSTM (The Elastic Stream)
        # ----------------------------------------------------------------------
        # Designed to act as a numerical integrator for volume-based pressure.
        self.lstm_branch = nn.LSTM(
            input_size=config.INPUT_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=config.DROPOUT if config.LSTM_LAYERS > 1 else 0,
        )
        self.lstm_out_dim = config.HIDDEN_DIM * 2

        # ----------------------------------------------------------------------
        # Fusion Head: Additive Residual Decomposition
        # ----------------------------------------------------------------------
        # Equation: P_pred = Head_TCN(H_tcn) + Head_LSTM(H_lstm) + Head_Joint([H_tcn, H_lstm])

        # 1. Resistive Head: Projects TCN output directly to scalar
        self.head_tcn = nn.Linear(self.tcn_out_dim, 1)

        # 2. Elastic Head: Projects LSTM output directly to scalar
        self.head_lstm = nn.Linear(self.lstm_out_dim, 1)

        # 3. Residual Head: Projects concatenation to scalar via MLP
        # Captures complex non-linear interactions that simple addition misses.
        self.head_joint = nn.Sequential(
            nn.Linear(self.tcn_out_dim + self.lstm_out_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """
        Forward pass of the PSA-Net.

        Args:
            x (torch.Tensor): Input features of shape (Batch, Seq_Len, Features).

        Returns:
            torch.Tensor: Predicted pressure of shape (Batch, Seq_Len).
        """
        # --- TCN Branch Forward ---
        # Permute to (Batch, Features, Seq_Len) for Conv1d
        x_tcn_in = x.transpose(1, 2)
        tcn_feat = self.tcn_branch(x_tcn_in)
        # Permute back to (Batch, Seq_Len, Channels)
        tcn_feat = tcn_feat.transpose(1, 2)

        # --- LSTM Branch Forward ---
        # LSTM expects (Batch, Seq_Len, Features)
        lstm_feat, _ = self.lstm_branch(x)

        # --- Additive Decomposition ---
        # 1. Calculate Resistive Pressure Component
        p_resistive = self.head_tcn(tcn_feat)

        # 2. Calculate Elastic Pressure Component
        p_elastic = self.head_lstm(lstm_feat)

        # 3. Calculate Interaction/Residual Component
        combined = torch.cat([tcn_feat, lstm_feat], dim=-1)
        p_interaction = self.head_joint(combined)

        # Sum components to get final pressure
        # Shape: (Batch, Seq_Len, 1)
        pressure = p_resistive + p_elastic + p_interaction

        # Squeeze the last dimension to match target shape (Batch, Seq_Len)
        return pressure.squeeze(-1)
