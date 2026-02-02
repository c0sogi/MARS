import torch
import torch.nn as nn
from library.config import Config


class TemporalCNN(nn.Module):
    """
    1D Convolutional Neural Network for NFL Contact Detection.

    This model processes a temporal window of player tracking features to predict
    contact events. It utilizes 1D convolutions to capture temporal patterns
    (velocity changes, approach angles) and Global Max Pooling to identify
    the most significant signal within the window.

    Architecture:
    - Input: (Batch, Channels, Time)
    - Conv Block 1: Conv1d -> BN -> ReLU
    - Conv Block 2: Conv1d -> BN -> ReLU
    - Global Max Pooling
    - Dropout
    - Fully Connected Layer
    - Sigmoid Activation
    """

    def __init__(self):
        super(TemporalCNN, self).__init__()

        # Hyperparameters from Config
        self.in_channels = Config.NUM_FEATURES
        self.hidden_channels = Config.HIDDEN_CHANNELS
        self.kernel_size = Config.KERNEL_SIZE
        self.dropout_p = Config.DROPOUT

        # Calculate padding to maintain temporal dimension (assuming odd kernel size)
        padding = self.kernel_size // 2

        # --- Feature Extraction (Temporal) ---

        # Block 1
        self.conv1 = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=padding,
            bias=False,  # Bias not needed with BatchNorm
        )
        self.bn1 = nn.BatchNorm1d(self.hidden_channels)

        # Block 2
        # Increasing channels to capture more complex combinations
        self.conv2 = nn.Conv1d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels * 2,
            kernel_size=self.kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(self.hidden_channels * 2)

        self.relu = nn.ReLU()

        # --- Aggregation ---
        # Global Max Pooling reduces (Batch, Channels, Time) -> (Batch, Channels, 1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # --- Classification Head ---
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.fc = nn.Linear(self.hidden_channels * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Time).

        Returns:
            torch.Tensor: Probability of contact, shape (Batch, 1).
        """
        # 1. Convolutional Blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # 2. Global Max Pooling
        # Shape: (Batch, Hidden*2, Time) -> (Batch, Hidden*2, 1)
        x = self.global_pool(x)

        # 3. Flatten
        # Shape: (Batch, Hidden*2)
        x = x.squeeze(-1)

        # 4. Classification Head
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x
