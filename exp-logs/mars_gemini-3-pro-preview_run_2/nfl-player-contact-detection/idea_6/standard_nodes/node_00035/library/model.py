import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class KinematicMLP(nn.Module):
    """
    Kinematic Multi-Layer Perceptron (K-MLP).

    A simple, high-capacity dense network designed to learn physics relationships
    from explicit relative kinematic features.
    Cite: solution_lesson_node_00023, solution_lesson_node_00033
    """

    def __init__(
        self,
        input_dim,
        center_dim,
        hidden_dim=Config.HIDDEN_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        super(KinematicMLP, self).__init__()

        # Combined input dimension (Wide Window + Center Frame)
        # We concatenate them early as the MLP can learn interactions across the window
        total_dim = input_dim + center_dim

        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_wide, x_center, condition):
        """
        Args:
            x_wide (torch.Tensor): Flattened window features.
            x_center (torch.Tensor): Features at t=0.
            condition (torch.Tensor): Unused in simple MLP (implicit learning).
        """
        # Concatenate all available features
        x = torch.cat([x_wide, x_center], dim=1)
        return self.net(x)
