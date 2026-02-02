import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class KinematicMLP(nn.Module):
    """
    Kinematic Multi-Layer Perceptron (K-MLP).

    Cite solution_lesson_node_00015: "Vectorized Lag-Shifting Outperforms Vertical Window Expansion".
    This model takes a flattened vector of windowed features (Wide Format) and processes it
    through a series of dense layers.
    """

    def __init__(
        self,
        input_dim=None,
        hidden_layers=None,
        dropout=None,
    ):
        super(KinematicMLP, self).__init__()

        # Cite debug_lesson_5: Defer Configuration Access to Runtime
        if input_dim is None:
            input_dim = Config.INPUT_DIM
        if hidden_layers is None:
            hidden_layers = Config.HIDDEN_LAYERS
        if dropout is None:
            dropout = Config.DROPOUT

        layers = []
        in_dim = input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x shape: (Batch, Window, Features) or (Batch, Window * Features)
        # We flatten if necessary
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = self.features(x)
        return torch.sigmoid(self.classifier(x))
