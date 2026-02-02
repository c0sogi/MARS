import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    INPUT_CHANNELS,
    HIDDEN_CHANNELS,
    KERNEL_SIZE,
    DROPOUT_RATE,
    WINDOW_SIZE,
)


class TemporalBlock(nn.Module):
    """
    A Residual Temporal Block containing two 1D Convolutional layers.
    Structure: Conv1d -> BN -> ReLU -> Dropout -> Conv1d -> BN -> ReLU -> Dropout
    Includes a residual connection.
    """

    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()

        # First convolution layer
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution layer
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Downsample layer for residual connection if dimensions match
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        return self.relu(out + residual)


class DualStreamTCN(nn.Module):
    """
    Dual-Stream Temporal Convolutional Network.

    Architecture:
    1. Temporal Backbone: Stack of TemporalBlocks to encode sequence data.
    2. Global Pooling: Aggregates temporal features.
    3. Dual Heads:
       - head_player: Detects player-player contact.
       - head_ground: Detects player-ground contact.
    4. Gating: Combines outputs based on 'is_ground' flag.
    """

    def __init__(self):
        super(DualStreamTCN, self).__init__()

        # Architecture Hyperparameters
        num_inputs = INPUT_CHANNELS
        # Progressively increase channels: 64 -> 128 -> 128
        num_channels = [HIDDEN_CHANNELS, HIDDEN_CHANNELS * 2, HIDDEN_CHANNELS * 2]
        kernel_size = KERNEL_SIZE
        dropout = DROPOUT_RATE

        # --- Temporal Backbone ---
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 1  # Fixed dilation for short window
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            # Padding to maintain sequence length
            padding = (kernel_size - 1) // 2

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )

        self.backbone = nn.Sequential(*layers)

        # --- Decision Heads ---
        embedding_dim = num_channels[-1]

        # Player-Player Contact Head
        self.head_player = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
        )

        # Player-Ground Contact Head
        self.head_ground = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(self, x, is_ground):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Window, Features)
            is_ground (torch.Tensor): Binary flag of shape (Batch,) or (Batch, 1)
                                      1 indicates checking for ground contact.

        Returns:
            torch.Tensor: Logits of shape (Batch, 1)
        """
        # Permute for Conv1d: (Batch, Window, Features) -> (Batch, Features, Window)
        x = x.permute(0, 2, 1)

        # 1. Temporal Encoding
        features = self.backbone(x)  # Output: (Batch, Hidden, Window)

        # 2. Global Max Pooling
        # Extracts the strongest feature activation across the time window (e.g., peak impact)
        embedding = torch.max(features, dim=2)[0]  # Output: (Batch, Hidden)

        # 3. Dual Head Prediction
        logits_player = self.head_player(embedding)  # (Batch, 1)
        logits_ground = self.head_ground(embedding)  # (Batch, 1)

        # 4. Gating Mechanism
        # Ensure is_ground is (Batch, 1) for broadcasting
        if is_ground.dim() == 1:
            is_ground = is_ground.unsqueeze(1)

        # Select the appropriate head output based on the interaction type
        final_logits = logits_ground * is_ground + logits_player * (1 - is_ground)

        return final_logits
