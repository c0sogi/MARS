import torch
import torch.nn as nn
from library.config import Config


class KETRNet(nn.Module):
    """
    Kinematically-Explicit Time-Resolved Network (KETR-Net).

    A 1D Convolutional Network designed for contact detection that strictly preserves
    temporal alignment and prioritizes the instantaneous state at the center frame.

    Architecture:
    1. Input: (Batch, Window_Size, Features)
    2. Conv1D Backbone: Extracts temporal patterns without pooling.
    3. Residual Connection: Concatenates flattened Conv output with raw Center Frame features.
    4. Head: Dense layers for classification.
    """

    def __init__(self, config: Config):
        super(KETRNet, self).__init__()

        self.window_size = config.window_size
        self.num_features = len(config.feature_cols)
        self.filters = config.cnn_filters
        self.kernel_size = config.cnn_kernel_size
        self.dropout_rate = config.dropout

        # Ensure window size is odd for a distinct center frame
        if self.window_size % 2 == 0:
            raise ValueError(f"Window size must be odd, got {self.window_size}")

        self.center_idx = self.window_size // 2

        # --- Time-Resolved Backbone ---
        # Stack of 1D Convolutions.
        # padding='same' ensures output length == input length (Window_Size).
        # No pooling layers are used to preserve temporal resolution.
        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels=self.num_features,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm1d(self.filters),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(
                in_channels=self.filters,
                out_channels=self.filters * 2,
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm1d(self.filters * 2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(
                in_channels=self.filters * 2,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm1d(self.filters),
            nn.ReLU(),
        )

        # --- Dimension Calculation ---
        # Flattened Conv Output Size = Filters * Window_Size
        self.flattened_conv_size = self.filters * self.window_size

        # Total Dense Input Size = Flattened Conv Output + Raw Center Features
        self.dense_input_size = self.flattened_conv_size + self.num_features

        # --- Unified Classification Head ---
        self.head = nn.Sequential(
            nn.Linear(self.dense_input_size, config.dense_hidden_units),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(config.dense_hidden_units, config.dense_hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(config.dense_hidden_units // 2, 1),  # Output logit
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Window_Size, Features)

        Returns:
            logits: Tensor of shape (Batch, 1)
        """
        batch_size = x.size(0)

        # 1. Extract Center Frame Features for Residual Connection
        # Shape: (Batch, Features)
        center_features = x[:, self.center_idx, :]

        # 2. Process through Conv Backbone
        # Permute for Conv1d: (Batch, Features, Window_Size)
        x_permuted = x.permute(0, 2, 1)

        # Apply Convolutions
        # Output Shape: (Batch, Filters, Window_Size)
        conv_out = self.conv_block(x_permuted)

        # 3. Flatten Temporal Features
        # Shape: (Batch, Filters * Window_Size)
        conv_flat = conv_out.view(batch_size, -1)

        # 4. Center-Frame Residual Connection
        # Concatenate learned temporal context with exact instantaneous state
        # Shape: (Batch, Filters*Window + Features)
        combined = torch.cat([conv_flat, center_features], dim=1)

        # 5. Classification Head
        logits = self.head(combined)

        return logits.squeeze(1)
