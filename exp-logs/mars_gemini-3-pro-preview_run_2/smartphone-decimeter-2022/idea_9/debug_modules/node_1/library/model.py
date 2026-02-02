import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    TRAJECTORY_INPUT_DIM,
    CONTEXT_INPUT_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    CNN_LAYERS,
    KERNEL_SIZE,
    DROPOUT_RATE,
)


class ResidualBlock1D(nn.Module):
    """
    A 1D Residual Block with Batch Normalization and Dropout.
    Structure: Conv1D -> BN -> ReLU -> Dropout -> Conv1D -> BN -> Add Input -> ReLU
    """

    def __init__(self, channels, kernel_size, dropout_rate):
        super(ResidualBlock1D, self).__init__()
        # Padding to maintain sequence length
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout_rate)

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
    Encodes the time-series trajectory data using a stack of Residual 1D CNN blocks.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, kernel_size, dropout_rate):
        super(TrajectoryEncoder, self).__init__()

        # Initial projection layer to scale up to hidden dimensions
        padding = kernel_size // 2
        self.initial_conv = nn.Conv1d(
            input_dim, hidden_dim, kernel_size, padding=padding, bias=False
        )
        self.initial_bn = nn.BatchNorm1d(hidden_dim)

        # Stack of residual blocks
        self.blocks = nn.ModuleList(
            [
                ResidualBlock1D(hidden_dim, kernel_size, dropout_rate)
                for _ in range(num_layers)
            ]
        )

        # Global Average Pooling to collapse the time dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x shape: [Batch, Input_Dim, Length]

        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = F.relu(x)

        for block in self.blocks:
            x = block(x)

        x = self.global_pool(x)
        x = x.squeeze(-1)  # Shape: [Batch, Hidden_Dim]
        return x


class ContextEncoder(nn.Module):
    """
    Encodes static environmental context features (satellite geometry) using an MLP.
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(ContextEncoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        # x shape: [Batch, Input_Dim]
        return self.net(x)


class GeometryConditionedCNN(nn.Module):
    """
    Main model architecture.
    Fuses the dynamic trajectory features with static environmental context
    to predict position residuals (Delta East, Delta North).
    """

    def __init__(self):
        super(GeometryConditionedCNN, self).__init__()

        self.traj_encoder = TrajectoryEncoder(
            input_dim=TRAJECTORY_INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=CNN_LAYERS,
            kernel_size=KERNEL_SIZE,
            dropout_rate=DROPOUT_RATE,
        )

        self.context_encoder = ContextEncoder(
            input_dim=CONTEXT_INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            dropout_rate=DROPOUT_RATE,
        )

        # Fusion Head
        # Concatenates the outputs of both encoders
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )

    def forward(self, traj, ctx):
        """
        Args:
            traj: Time-series input [Batch, Channels, Length]
            ctx: Static context input [Batch, Features]
        Returns:
            output: Predicted residuals [Batch, 2]
        """
        traj_feat = self.traj_encoder(traj)
        ctx_feat = self.context_encoder(ctx)

        combined = torch.cat([traj_feat, ctx_feat], dim=1)
        output = self.head(combined)

        return output
