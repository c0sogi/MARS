import torch
import torch.nn as nn
from library.config import Config


class KinematicMLP(nn.Module):
    """
    Kinematic Multi-Layer Perceptron (K-MLP).

    A simple, robust architecture for tabular kinematic data.
    Cite Lesson 00023: Simplicity Enables Data Scale.
    """

    def __init__(self):
        super(KinematicMLP, self).__init__()

        input_dim = Config.INPUT_DIM
        hidden_layers = Config.HIDDEN_LAYERS
        dropout_rate = Config.DROPOUT_RATE

        layers = []
        in_dim = input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        # Final projection
        layers.append(nn.Linear(in_dim, 1))

        self.model = nn.Sequential(*layers)
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
        return self.model(x)
