import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    WINDOW_SIZE,
    KINEMATIC_FEATURES,
    CNN_HIDDEN_DIM,
    CNN_KERNEL_SIZE,
    CNN_LAYERS,
    CNN_DROPOUT,
    MLP_HIDDEN_DIM,
    MLP_DROPOUT,
)


class ResidualBlock1D(nn.Module):
    """
    A 1D Residual Block with two convolution layers.
    """

    def __init__(self, channels, kernel_size, dropout):
        super(ResidualBlock1D, self).__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)
        return out


class KinematicCNN(nn.Module):
    """
    Shallow 1D-CNN for processing kinematic sequences.
    Preserves temporal structure by flattening instead of pooling.
    """

    def __init__(
        self, input_channels, seq_length, hidden_dim, kernel_size, num_layers, dropout
    ):
        super(KinematicCNN, self).__init__()

        # Initial expansion
        padding = kernel_size // 2
        self.input_conv = nn.Conv1d(
            input_channels, hidden_dim, kernel_size, padding=padding
        )
        self.input_bn = nn.BatchNorm1d(hidden_dim)

        # Residual layers
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock1D(hidden_dim, kernel_size, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output dimension calculation (Flattened)
        self.output_dim = hidden_dim * seq_length

    def forward(self, x):
        # x shape: (Batch, Channels, Length)
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)

        for block in self.res_blocks:
            x = block(x)

        # Flatten: (Batch, Hidden * Length)
        x = x.view(x.size(0), -1)
        return x


class SkyContextMLP(nn.Module):
    """
    MLP for embedding sky context statistics.
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super(SkyContextMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        return self.net(x)


class SCRCNN(nn.Module):
    """
    Sky-Contextualized Relative-Motion CNN.
    Fuses kinematic trajectory features with environmental context.
    """

    def __init__(self):
        super(SCRCNN, self).__init__()

        # 1. Kinematic Stream
        # Input channels = number of kinematic features
        kin_input_channels = len(KINEMATIC_FEATURES)
        self.kinematic_net = KinematicCNN(
            input_channels=kin_input_channels,
            seq_length=WINDOW_SIZE,
            hidden_dim=CNN_HIDDEN_DIM,
            kernel_size=CNN_KERNEL_SIZE,
            num_layers=CNN_LAYERS,
            dropout=CNN_DROPOUT,
        )

        # 2. Sky Context Stream
        # Input dim = 6 (Mean/Std for Elevation, Azimuth, Cn0)
        # Based on preprocessing.py aggregation logic
        sky_input_dim = 6
        self.sky_net = SkyContextMLP(
            input_dim=sky_input_dim, hidden_dim=MLP_HIDDEN_DIM, dropout=MLP_DROPOUT
        )

        # 3. Fusion Head
        fusion_input_dim = self.kinematic_net.output_dim + self.sky_net.output_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_input_dim, MLP_HIDDEN_DIM),
            nn.BatchNorm1d(MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(MLP_HIDDEN_DIM, MLP_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(MLP_HIDDEN_DIM // 2, 2),  # Output: dLat_m, dLon_m
        )

    def forward(self, x_kin, x_sky):
        """
        Args:
            x_kin: (Batch, Channels, Length)
            x_sky: (Batch, Features)
        """
        # Process branches
        feat_kin = self.kinematic_net(x_kin)
        feat_sky = self.sky_net(x_sky)

        # Fuse
        combined = torch.cat([feat_kin, feat_sky], dim=1)

        # Predict
        out = self.head(combined)
        return out
