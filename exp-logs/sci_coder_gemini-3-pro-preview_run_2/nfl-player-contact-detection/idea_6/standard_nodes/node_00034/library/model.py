import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class KinematicMLP(nn.Module):
    """
    Kinematic Multi-Layer Perceptron (K-MLP).

    A simple, robust architecture that learns from a flattened temporal window of
    explicitly engineered kinematic features.
    Cite Lesson 00023: Simplicity Enables Data Scale.
    Cite Lesson 00033: Explicit Kinematics Trumps Architectural Conditioning.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        super(KinematicMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
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

    def forward(self, x):
        return self.net(x)
