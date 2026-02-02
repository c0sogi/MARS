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


class KinematicMLP(nn.Module):
    """
    Kinematic Multi-Layer Perceptron (K-MLP).

    Processes the entire temporal window as a flattened feature vector.
    Preserves temporal alignment by avoiding global pooling.
    Cite solution_lesson_node_00011
    """

    def __init__(self):
        super(KinematicMLP, self).__init__()

        # Flattened input dimension: Window * Features
        input_dim = WINDOW_SIZE * INPUT_CHANNELS

        # MLP Architecture
        # 512 -> 256 -> 128 -> 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 1),
        )

    def forward(self, x, is_ground=None):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Window, Features)
            is_ground: Unused in MLP (included as feature in x), kept for compatibility.

        Returns:
            torch.Tensor: Logits of shape (Batch, 1)
        """
        batch_size = x.size(0)

        # Flatten the temporal window
        # (Batch, Window, Features) -> (Batch, Window * Features)
        x_flat = x.view(batch_size, -1)

        return self.net(x_flat)
