import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    TRAJECTORY_FEATURES,
    SKY_FEATURES,
    TARGET_FEATURES,
    CNN_CHANNELS,
    CNN_KERNEL_SIZE,
    CNN_LAYERS,
    CNN_DROPOUT,
    SKY_HIDDEN_DIM,
    SKY_DROPOUT,
    FUSION_HIDDEN_DIM,
    FUSION_DROPOUT,
    WINDOW_SIZE,
)


class ResidualBlock1D(nn.Module):
    """
    A 1D Residual Block with Batch Normalization and Dropout.
    Structure: Conv1d -> BN -> ReLU -> Dropout -> Conv1d -> BN -> Add Input -> ReLU
    """

    def __init__(self, channels, kernel_size, dropout):
        super(ResidualBlock1D, self).__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
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


class TrajectoryEncoder(nn.Module):
    """
    Encodes the time-series trajectory window using a 1D CNN backbone.
    Input: (Batch, Channels, Length)
    Output: (Batch, Hidden_Dim)
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, dropout):
        super(TrajectoryEncoder, self).__init__()

        # Initial projection
        # We use a kernel size of 1 here to project features independently per timestep first,
        # or a larger kernel to capture immediate context. Let's use kernel_size.
        padding = kernel_size // 2
        self.input_conv = nn.Conv1d(
            input_dim, hidden_dim, kernel_size, padding=padding, bias=False
        )
        self.input_bn = nn.BatchNorm1d(hidden_dim)

        # Stack of residual blocks
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock1D(hidden_dim, kernel_size, dropout)
                for _ in range(num_layers)
            ]
        )

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x shape: (Batch, Input_Dim, Window_Size)
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)

        for block in self.res_blocks:
            x = block(x)

        # Pooling: (Batch, Hidden_Dim, Window_Size) -> (Batch, Hidden_Dim, 1)
        x = self.global_pool(x)

        # Flatten: (Batch, Hidden_Dim)
        x = x.squeeze(-1)
        return x


class SkyContextEncoder(nn.Module):
    """
    Encodes the static satellite geometry statistics using an MLP.
    Input: (Batch, Input_Dim)
    Output: (Batch, Hidden_Dim)
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super(SkyContextEncoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class SkyMotionModel(nn.Module):
    """
    Dual-stream model fusing Trajectory and Sky Context streams to predict residuals.
    """

    def __init__(self):
        super(SkyMotionModel, self).__init__()

        # 1. Trajectory Stream
        traj_input_dim = len(TRAJECTORY_FEATURES)
        self.traj_encoder = TrajectoryEncoder(
            input_dim=traj_input_dim,
            hidden_dim=CNN_CHANNELS,
            kernel_size=CNN_KERNEL_SIZE,
            num_layers=CNN_LAYERS,
            dropout=CNN_DROPOUT,
        )

        # 2. Sky Context Stream
        sky_input_dim = len(SKY_FEATURES)
        self.sky_encoder = SkyContextEncoder(
            input_dim=sky_input_dim, hidden_dim=SKY_HIDDEN_DIM, dropout=SKY_DROPOUT
        )

        # 3. Fusion Head
        fusion_input_dim = CNN_CHANNELS + SKY_HIDDEN_DIM
        output_dim = len(TARGET_FEATURES)  # 2: d_lat_m, d_lon_m

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, FUSION_HIDDEN_DIM),
            nn.BatchNorm1d(FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(FUSION_DROPOUT),
            nn.Linear(FUSION_HIDDEN_DIM, FUSION_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(FUSION_HIDDEN_DIM // 2, output_dim),
        )

    def forward(self, traj, sky):
        """
        Args:
            traj: (Batch, Traj_Channels, Window_Size)
            sky: (Batch, Sky_Channels)
        Returns:
            out: (Batch, 2) -> Predicted residuals in meters
        """
        traj_feat = self.traj_encoder(traj)
        sky_feat = self.sky_encoder(sky)

        combined = torch.cat([traj_feat, sky_feat], dim=1)
        out = self.fusion_head(combined)

        return out
