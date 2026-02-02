import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TemporalBlock(nn.Module):
    """
    A single residual block for the TCN backbone.
    Uses 1D Convolutions with padding to maintain sequence length.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # For 'same' padding with dilation
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.bn2,
            self.relu2,
            self.dropout2,
        )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (Batch, Channels, Seq_Len)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalBackbone(nn.Module):
    """
    TCN Backbone consisting of stacked TemporalBlocks.
    """

    def __init__(
        self, input_dim, num_channels, kernel_size=3, num_layers=4, dropout=0.2
    ):
        super(TemporalBackbone, self).__init__()
        layers = []
        num_levels = num_layers

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_dim if i == 0 else num_channels
            out_channels = num_channels

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation_size,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Input: (Batch, Seq_Len, Features)
        # Conv1d expects (Batch, Channels, Seq_Len)
        x = x.permute(0, 2, 1)
        y = self.network(x)
        # Permute back to (Batch, Seq_Len, Channels)
        y = y.permute(0, 2, 1)
        return y


class GnssModel(nn.Module):
    """
    Simplified TCN for GNSS location prediction using aggregated features.

    Architecture:
    1. Temporal Backbone (TCN): Process sequence of aggregated features
    2. Output Head: Predict residuals (Delta Lat, Delta Lon)
    """

    def __init__(self):
        super(GnssModel, self).__init__()

        # Temporal Backbone
        self.tcn = TemporalBackbone(
            input_dim=Config.TCN_INPUT_DIM,
            num_channels=Config.TCN_CHANNELS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            num_layers=Config.TCN_LAYERS,
            dropout=Config.TCN_DROPOUT,
        )

        # Output Head
        self.head = nn.Sequential(
            nn.Linear(Config.TCN_CHANNELS, 32),
            nn.ReLU(),
            nn.Linear(32, Config.OUTPUT_DIM),  # Predicts (dLat, dLon)
        )

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq_Len, Features)

        Returns:
            residuals: (Batch, Seq_Len, 2)
        """
        # Temporal Processing
        temporal_feats = self.tcn(x)

        # Prediction
        residuals = self.head(temporal_feats)

        return residuals
