import torch
import torch.nn as nn
from library.config import Config


class K_MLP(nn.Module):
    """
    Kinematic Multi-Layer Perceptron (K-MLP).
    Simple, deep dense network that outperforms complex architectures on this dataset
    due to resolution limits (Cite solution_lesson_node_00059).
    """

    def __init__(
        self,
        input_dim,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
    ):
        super(K_MLP, self).__init__()

        layers = []
        current_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_size

        layers.append(nn.Linear(current_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
