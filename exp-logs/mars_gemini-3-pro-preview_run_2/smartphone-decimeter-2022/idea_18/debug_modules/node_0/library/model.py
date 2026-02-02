import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock1D(nn.Module):
    """
    A residual block with 1D convolutions, Batch Normalization, and ReLU activation.
    Maintains the temporal dimension size (padding='same').
    """

    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.0):
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


class KinematicEncoder(nn.Module):
    """
    Processes the sliding window of kinematic data using a shallow 1D-CNN.
    Crucially, it flattens the output instead of pooling to preserve temporal structure
    relative to the window center.
    """

    def __init__(self):
        super(KinematicEncoder, self).__init__()

        in_channels = Config.CNN_IN_CHANNELS
        hidden_channels = Config.CNN_HIDDEN_CHANNELS
        kernel_size = Config.CNN_KERNEL_SIZE
        num_layers = Config.CNN_LAYERS
        dropout = Config.CNN_DROPOUT

        # Initial projection to hidden dimension
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        # Stack of residual blocks
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock1D(hidden_channels, kernel_size, dropout)
                for _ in range(num_layers)
            ]
        )

        # Calculate output dimension after flattening
        # Shape: (Batch, Hidden, Window) -> (Batch, Hidden * Window)
        self.output_dim = hidden_channels * Config.WINDOW_SIZE

    def forward(self, x):
        # x shape: (Batch, In_Channels, Window_Size)
        x = self.input_proj(x)

        for block in self.res_blocks:
            x = block(x)

        # Flatten: (Batch, Hidden, Window) -> (Batch, Hidden * Window)
        x = x.view(x.size(0), -1)
        return x


class SkyContextEncoder(nn.Module):
    """
    Processes the aggregated sky state statistics using a simple MLP.
    """

    def __init__(self):
        super(SkyContextEncoder, self).__init__()

        in_dim = Config.SKY_IN_DIM
        hidden_dim = Config.SKY_HIDDEN_DIM

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.output_dim = hidden_dim

    def forward(self, x):
        # x shape: (Batch, Sky_In_Dim)
        return self.net(x)


class FusionHead(nn.Module):
    """
    Fuses the kinematic and sky embeddings and predicts the 2D metric residuals.
    """

    def __init__(self, input_dim: int):
        super(FusionHead, self).__init__()

        hidden_dims = Config.FUSION_HIDDEN_DIMS
        output_dim = Config.OUTPUT_DIM

        layers = []
        curr_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SKFNet(nn.Module):
    """
    Sky-Conditioned Kinematic Filtering Network (SKF-Net).
    Combines a 1D-CNN for trajectory smoothing with an MLP for environmental bias correction.
    """

    def __init__(self):
        super(SKFNet, self).__init__()

        self.kinematic_encoder = KinematicEncoder()
        self.sky_encoder = SkyContextEncoder()

        fusion_input_dim = (
            self.kinematic_encoder.output_dim + self.sky_encoder.output_dim
        )
        self.head = FusionHead(fusion_input_dim)

    def forward(self, x_seq, x_sky):
        """
        Args:
            x_seq: Kinematic sequence tensor (Batch, Channels, Window)
            x_sky: Sky context tensor (Batch, Features)

        Returns:
            out: Predicted residuals (Batch, 2) -> (DeltaEast, DeltaNorth)
        """
        # Encode streams
        kinematic_emb = self.kinematic_encoder(x_seq)
        sky_emb = self.sky_encoder(x_sky)

        # Concatenate
        fused = torch.cat([kinematic_emb, sky_emb], dim=1)

        # Predict
        out = self.head(fused)
        return out
