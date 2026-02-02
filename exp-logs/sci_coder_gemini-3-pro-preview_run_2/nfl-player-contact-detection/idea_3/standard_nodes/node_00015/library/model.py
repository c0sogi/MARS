import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TRGCN(nn.Module):
    """
    Time-Resolved Gated Convolutional Network (TR-GCN).

    This architecture combines a 1D Convolutional encoder (without pooling) to capture
    temporal patterns while preserving resolution, with a gated dual-head output
    that applies specific logic for Ground vs. Player contacts.
    """

    def __init__(
        self,
        input_dim=Config.NUM_FEATURES_PER_TIMESTEP,
        window_size=Config.WINDOW_SIZE,
        cnn_filters=Config.CNN_FILTERS,
        kernel_size=Config.CNN_KERNEL_SIZE,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT,
    ):
        """
        Args:
            input_dim (int): Number of features per timestep.
            window_size (int): Temporal window size (number of frames).
            cnn_filters (int): Number of filters in Conv1d layers.
            kernel_size (int): Kernel size for Conv1d layers.
            hidden_dim (int): Dimension of the shared dense layer.
            dropout (float): Dropout probability.
        """
        super(TRGCN, self).__init__()

        self.window_size = window_size

        # 1D Convolutional Encoder
        # We use padding to preserve the temporal dimension (Same Padding)
        # No Global Pooling is used to retain temporal position information
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_filters,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(cnn_filters)

        self.conv2 = nn.Conv1d(
            in_channels=cnn_filters,
            out_channels=cnn_filters,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm1d(cnn_filters)

        # Flattening
        # The output of convs is (Batch, Filters, Window_Size)
        # We flatten this to preserve the structure: "Feature F at Time T"
        self.flatten_dim = cnn_filters * window_size

        # Shared Dense Section
        self.fc_shared = nn.Linear(self.flatten_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

        # Dual-Head Output
        # Head 1: Specialized for Player-to-Player contact
        self.head_player = nn.Linear(hidden_dim, 1)

        # Head 2: Specialized for Player-to-Ground contact
        self.head_ground = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Forward pass with Gating Mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Window_Size, Features).
                              The last feature is expected to be 'is_ground'.

        Returns:
            torch.Tensor: Probability of contact (Batch, 1).
        """
        # 1. Extract Gating Signal
        # is_ground is the last feature. We take the value from the center frame.
        # Shape: (Batch, 1)
        center_idx = self.window_size // 2
        is_ground = x[:, center_idx, -1].unsqueeze(1)

        # 2. Permute for Conv1d
        # Input: (Batch, Window, Features) -> Output: (Batch, Features, Window)
        x = x.permute(0, 2, 1)

        # 3. Convolutional Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # 4. Flatten
        # (Batch, Filters, Window) -> (Batch, Filters * Window)
        x = x.view(x.size(0), -1)

        # 5. Shared Dense Layer
        x = self.fc_shared(x)
        x = F.relu(x)
        x = self.dropout_layer(x)

        # 6. Dual Heads (Probabilities)
        # We use Sigmoid here to generate probabilities for the gating logic
        prob_player = torch.sigmoid(self.head_player(x))
        prob_ground = torch.sigmoid(self.head_ground(x))

        # 7. Gating Mechanism
        # Deterministically select the output based on the physics type (Ground vs Player)
        # P_final = P_ground * I_ground + P_player * (1 - I_ground)
        final_prob = prob_ground * is_ground + prob_player * (1.0 - is_ground)

        return final_prob
