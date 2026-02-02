import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class KinematicMLP(nn.Module):
    """
    Kinematic Multi-Layer Perceptron (K-MLP).

    Processes a flattened vector of windowed tracking features.
    Cite solution_lesson_node_00011: Avoids global pooling to preserve temporal alignment.
    """

    def __init__(
        self,
        input_dim,
        hidden_layers=Config.MLP_HIDDEN_LAYERS,
        dropout=Config.DROPOUT,
    ):
        super(KinematicMLP, self).__init__()

        layers = []
        curr_dim = input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x is (Batch, Input_Dim)
        return torch.sigmoid(self.network(x))
