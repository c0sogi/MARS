import torch
import torch.nn as nn
from library.config import INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM


class ResidualMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for predicting GNSS position residuals.

    This model takes a flattened vector of satellite features as input and outputs
    a 3-dimensional vector representing the correction (residual) needed for the
    WLS baseline position in ECEF coordinates (dx, dy, dz).

    Architecture:
    - Input Layer: Linear transformation from input_dim to first hidden_dim.
    - Hidden Layers: Sequence of (Linear -> BatchNorm -> ReLU).
    - Output Layer: Linear transformation to output_dim (3).
    """

    def __init__(
        self, input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, output_dim=OUTPUT_DIM
    ):
        """
        Initialize the MLP.

        Args:
            input_dim (int): Dimension of the input feature vector. Defaults to config.INPUT_DIM.
            hidden_dims (list of int): List of hidden layer dimensions. Defaults to config.HIDDEN_DIMS.
            output_dim (int): Dimension of the output vector. Defaults to config.OUTPUT_DIM.
        """
        super(ResidualMLP, self).__init__()

        layers = []
        curr_dim = input_dim

        # Build hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim

        # Output layer
        layers.append(nn.Linear(curr_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize layer weights using Kaiming Normal initialization for Linear layers
        and standard initialization for BatchNorm layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        return self.network(x)
