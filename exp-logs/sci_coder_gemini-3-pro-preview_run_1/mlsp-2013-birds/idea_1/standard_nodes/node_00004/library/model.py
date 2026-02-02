import torch
import torch.nn as nn
from library.config import Config


class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for multi-label bird species classification.

    Architecture:
    - Input Layer: Accepts fixed-length feature vectors (default 100-dim).
    - Hidden Layers: Fully connected layers with ReLU activation and Dropout.
    - Output Layer: Linear projection to class logits (default 19 classes).

    The forward pass returns raw logits to be used with BCEWithLogitsLoss for numerical stability.
    """

    def __init__(
        self,
        input_dim=Config.INPUT_DIM,
        num_classes=Config.NUM_CLASSES,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        """
        Initializes the MLP model.

        Args:
            input_dim (int): Dimensionality of the input feature vector.
            num_classes (int): Number of output classes (species).
            hidden_layers (list): List of integers defining the size of each hidden layer.
            dropout_rate (float): Probability of an element to be zeroed in Dropout layers.
        """
        super(SimpleMLP, self).__init__()

        layers = []
        current_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        return self.network(x)
