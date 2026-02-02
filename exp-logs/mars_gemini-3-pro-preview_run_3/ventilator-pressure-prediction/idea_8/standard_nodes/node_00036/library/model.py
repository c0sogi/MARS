import torch
import torch.nn as nn
from library.config import Config


class PSANet(nn.Module):
    """
    Context-Aware Parallel Hybrid Network (CAP-Net).

    Optimized based on lessons:
    1. Uses Concatenation Fusion instead of Additive Decomposition (Cite solution_lesson_node_00034).
    2. Uses Dilated Convolutions in TCN branch (Cite solution_lesson_node_00035).
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # ----------------------------------------------------------------------
        # Branch 1: Dilated TCN (The Resistive Stream)
        # ----------------------------------------------------------------------
        # Uses dilation to expand receptive field exponentially.
        tcn_layers = []
        in_channels = config.INPUT_DIM
        kernel_size = config.TCN_KERNEL_SIZE

        for i, out_channels in enumerate(config.TCN_CHANNELS):
            dilation = 2**i
            # Calculate padding to maintain sequence length: P = d * (k-1) / 2
            padding = (dilation * (kernel_size - 1)) // 2

            tcn_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
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
        # Fusion Head: Concatenation
        # ----------------------------------------------------------------------
        # Fuses features via concatenation + MLP (Cite solution_lesson_node_00034)
        self.head = nn.Sequential(
            nn.Linear(self.tcn_out_dim + self.lstm_out_dim, 256),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        # --- TCN Branch Forward ---
        x_tcn_in = x.transpose(1, 2)
        tcn_feat = self.tcn_branch(x_tcn_in)
        tcn_feat = tcn_feat.transpose(1, 2)

        # --- LSTM Branch Forward ---
        lstm_feat, _ = self.lstm_branch(x)

        # --- Concatenation Fusion ---
        combined = torch.cat([tcn_feat, lstm_feat], dim=-1)
        pressure = self.head(combined)

        return pressure.squeeze(-1)
