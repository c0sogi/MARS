import torch
import torch.nn as nn
from library.config import HIDDEN_DIMS, DROPOUT_RATE, SEED, setup_reproducibility

# Set seeds for reproducibility
setup_reproducibility(SEED)


class ContactMLP(nn.Module):
    """
    A Time-Windowed Multi-Layer Perceptron (MLP) for contact detection.

    This architecture processes a flattened vector of kinematic features over a
    temporal window. It consists of a sequence of dense layers with Batch Normalization,
    ReLU activation, and Dropout, culminating in a Sigmoid output for binary classification.
    """

    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, dropout_rate=DROPOUT_RATE):
        """
        Initialize the MLP model.

        Args:
            input_dim (int): The size of the flattened input feature vector.
            hidden_dims (list of int): A list containing the number of neurons for each hidden layer.
                                       Defaults to configuration in library.config.
            dropout_rate (float): The dropout probability.
                                  Defaults to configuration in library.config.
        """
        super(ContactMLP, self).__init__()

        layers = []
        current_dim = input_dim

        # Build hidden layers dynamically based on configuration
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        # Output layer: Project to scalar and apply Sigmoid for probability
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output probabilities of shape (batch_size, 1).
        """
        return self.network(x)
