import torch
import torch.nn as nn
from library.config import Config


class KinematicMLP(nn.Module):
    """
    Simple Kinematic Multi-Layer Perceptron.
    Cite solution_lesson_node_00033: Explicit Kinematics Trumps Architectural Complexity.
    """

    def __init__(
        self,
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        super(KinematicMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x):
        return self.net(x)
