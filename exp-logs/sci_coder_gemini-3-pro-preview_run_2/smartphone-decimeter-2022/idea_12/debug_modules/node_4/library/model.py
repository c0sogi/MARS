import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock1D(nn.Module):
    """
    A 1D Residual Block with Batch Normalization and ReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        super(ResidualBlock1D, self).__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        # Shortcut connection to match dimensions if necessary
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

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
    Encodes the sliding window of GNSS/Motion features using 1D Convolutions.
    Outputs a flattened vector to preserve temporal structure relative to the window center.
    """

    def __init__(self, in_channels, window_size, hidden_dim, dropout=0.0):
        super(TrajectoryEncoder, self).__init__()

        # Stack of residual blocks increasing in depth
        # Block 1: Input -> Hidden
        self.block1 = ResidualBlock1D(
            in_channels, hidden_dim, kernel_size=3, dropout=dropout
        )

        # Block 2: Hidden -> Hidden * 2
        self.block2 = ResidualBlock1D(
            hidden_dim, hidden_dim * 2, kernel_size=3, dropout=dropout
        )

        # Block 3: Hidden * 2 -> Hidden * 2 (Refinement)
        self.block3 = ResidualBlock1D(
            hidden_dim * 2, hidden_dim * 2, kernel_size=3, dropout=dropout
        )

        # Calculate the size of the flattened output
        # We do not use pooling, so the sequence length (window_size) is preserved
        self.flatten_dim = (hidden_dim * 2) * window_size

    def forward(self, x):
        # x shape: (Batch, Channels, Window_Size)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Flatten: (Batch, Channels * Window_Size)
        x = x.flatten(1)
        return x


class SkyContextEncoder(nn.Module):
    """
    Encodes static environmental statistics (satellite geometry) using an MLP.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.0):
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
        # x shape: (Batch, Context_Dim)
        return self.net(x)


class SkyContextualizedCNN(nn.Module):
    """
    Dual-stream network combining trajectory dynamics and sky context to predict
    position residuals (d_east, d_north).
    """

    def __init__(self):
        super(SkyContextualizedCNN, self).__init__()

        # 1. Trajectory Stream
        self.traj_encoder = TrajectoryEncoder(
            in_channels=Config.TRAJECTORY_CHANNELS,
            window_size=Config.WINDOW_SIZE,
            hidden_dim=Config.CNN_HIDDEN_DIM,
            dropout=Config.DROPOUT_RATE,
        )

        # 2. Sky Context Stream
        self.sky_encoder = SkyContextEncoder(
            input_dim=Config.CONTEXT_INPUT_DIM,
            hidden_dim=Config.MLP_HIDDEN_DIM,
            dropout=Config.DROPOUT_RATE,
        )

        # 3. Fusion Head
        # Concatenate flattened trajectory features and sky embedding
        fusion_input_dim = self.traj_encoder.flatten_dim + Config.MLP_HIDDEN_DIM

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # Output: d_east, d_north (meters)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, traj, sky):
        """
        Args:
            traj: (Batch, Channels, Window_Size)
            sky: (Batch, Context_Dim)
        Returns:
            output: (Batch, 2) -> [d_east, d_north]
        """
        # Encode trajectory
        traj_emb = self.traj_encoder(traj)

        # Encode sky context
        sky_emb = self.sky_encoder(sky)

        # Fuse
        fused = torch.cat([traj_emb, sky_emb], dim=1)

        # Predict residuals
        output = self.fusion_head(fused)

        return output
