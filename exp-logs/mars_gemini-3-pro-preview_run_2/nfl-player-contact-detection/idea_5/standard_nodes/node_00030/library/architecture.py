import torch
import torch.nn as nn
from library.config import Config


class KinematicMLP(nn.Module):
    """
    Kinematic Multi-Layer Perceptron (K-MLP).
    Cite solution_lesson_node_00023: Simplicity Enables Data Scale.

    A simple but deep MLP that processes the flattened window of kinematic features.
    """

    def __init__(self):
        super(KinematicMLP, self).__init__()

        input_dim = Config.INPUT_WIDTH
        layers = Config.HIDDEN_LAYERS
        dropout = Config.DROPOUT

        # Build MLP
        net_layers = []

        # Input -> Hidden 1
        net_layers.append(nn.Linear(input_dim, layers[0]))
        net_layers.append(nn.BatchNorm1d(layers[0]))
        net_layers.append(nn.ReLU())
        net_layers.append(nn.Dropout(dropout))

        # Hidden -> Hidden
        for i in range(len(layers) - 1):
            net_layers.append(nn.Linear(layers[i], layers[i + 1]))
            net_layers.append(nn.BatchNorm1d(layers[i + 1]))
            net_layers.append(nn.ReLU())
            net_layers.append(nn.Dropout(dropout))

        # Output Head
        net_layers.append(nn.Linear(layers[-1], 1))

        self.net = nn.Sequential(*net_layers)
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
