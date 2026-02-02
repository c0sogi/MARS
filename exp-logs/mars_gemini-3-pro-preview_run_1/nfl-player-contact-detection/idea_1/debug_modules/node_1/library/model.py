import torch
import torch.nn as nn
from library.config import Config


class KinematicFFN(nn.Module):
    """
    Kinematic Feed-Forward Network (K-FFN).

    A lightweight Multi-Layer Perceptron (MLP) designed to classify contact events
    based solely on instantaneous tracking data.

    Architecture:
    - Input Linear Layer
    - Hidden Layers (defined by Config.HIDDEN_LAYERS, e.g., 64, 32 units)
      equipped with ReLU activation, Batch Normalization, and Dropout.
    - Output Layer: 1 unit with Sigmoid activation.
    """

    def __init__(self, input_dim, hidden_layers=None, dropout_rate=None):
        """
        Initialize the K-FFN model.

        Args:
            input_dim (int): Dimensionality of the input feature vector.
            hidden_layers (list, optional): List of integers defining the size of hidden layers.
                                            Defaults to Config.HIDDEN_LAYERS.
            dropout_rate (float, optional): Dropout probability for regularization.
                                            Defaults to Config.DROPOUT_RATE.
        """
        super(KinematicFFN, self).__init__()

        # Use defaults from Config if arguments are not explicitly provided
        if hidden_layers is None:
            hidden_layers = Config.HIDDEN_LAYERS
        if dropout_rate is None:
            dropout_rate = Config.DROPOUT_RATE

        layers = []
        current_dim = input_dim

        # Build hidden layers dynamically based on the configuration
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        # Output layer: Linear projection to 1 unit followed by Sigmoid
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) representing
                          the probability of contact.
        """
        return self.network(x)
