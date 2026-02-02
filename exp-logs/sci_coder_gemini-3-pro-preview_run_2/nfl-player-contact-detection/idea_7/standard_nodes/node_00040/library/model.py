import torch
import torch.nn as nn
from library.config import Config


class KinematicMLP(nn.Module):
    """
    Simple Kinematic MLP for Wide-Format Features.
    Cite: Lesson 00023 (Simplicity Enables Data Scale)
    """

    def __init__(self, config: Config):
        super(KinematicMLP, self).__init__()

        self.hidden_units = config.hidden_units
        self.dropout_rate = config.dropout

        layers = []
        # Input -> Hidden 1
        # Using LazyLinear to infer input dimension from the first batch
        layers.append(nn.LazyLinear(self.hidden_units[0]))
        layers.append(nn.BatchNorm1d(self.hidden_units[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))

        # Hidden 1 -> Hidden 2
        layers.append(nn.Linear(self.hidden_units[0], self.hidden_units[1]))
        layers.append(nn.BatchNorm1d(self.hidden_units[1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))

        # Hidden 2 -> Hidden 3
        layers.append(nn.Linear(self.hidden_units[1], self.hidden_units[2]))
        layers.append(nn.BatchNorm1d(self.hidden_units[2]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))

        # Output
        layers.append(nn.Linear(self.hidden_units[2], 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch, Features)
        return self.model(x).squeeze(1)
