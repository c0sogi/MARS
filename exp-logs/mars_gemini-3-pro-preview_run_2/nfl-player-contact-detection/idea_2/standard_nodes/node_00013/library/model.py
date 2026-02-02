import torch
import torch.nn as nn
from library.config import (
    INPUT_CHANNELS,
    HIDDEN_CHANNELS,
    DROPOUT_RATE,
    WINDOW_SIZE,
)


class KinematicMLP(nn.Module):
    """
    Kinematic Multi-Layer Perceptron (K-MLP).

    Flattens the temporal window of features into a single vector and processes
    it through a series of dense layers. This preserves the absolute temporal
    position of each feature (e.g., t-5 vs t=0 vs t+5), avoiding the loss of
    temporal alignment caused by global pooling.
    """

    def __init__(self):
        super(KinematicMLP, self).__init__()

        # Input dimension is Window * Features per frame
        input_dim = WINDOW_SIZE * INPUT_CHANNELS

        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_CHANNELS),
            nn.BatchNorm1d(HIDDEN_CHANNELS),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_CHANNELS, HIDDEN_CHANNELS // 2),
            nn.BatchNorm1d(HIDDEN_CHANNELS // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_CHANNELS // 2, 1),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Window, Features)

        Returns:
            torch.Tensor: Logits of shape (Batch, 1)
        """
        # Flatten: (Batch, Window, Features) -> (Batch, Window * Features)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        return self.net(x_flat)
