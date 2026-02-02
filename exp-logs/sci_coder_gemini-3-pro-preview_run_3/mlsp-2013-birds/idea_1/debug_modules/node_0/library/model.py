import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class ShallowMLP(nn.Module):
    """
    Shallow Multi-Layer Perceptron for Bird Species Classification.

    Architecture:
    Input -> Linear -> ReLU -> Dropout -> Linear -> Output (Logits)
    """

    def __init__(
        self,
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
    ):
        """
        Initializes the ShallowMLP model.

        Args:
            input_dim (int): Dimension of input features. Default from config.
            hidden_dim (int): Dimension of the hidden layer. Default from config.
            num_classes (int): Number of output classes. Default from config.
            dropout_rate (float): Dropout probability. Default from config.
        """
        super(ShallowMLP, self).__init__()

        # First dense layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout_rate)

        # Output dense layer
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        # Linear transformation
        x = self.fc1(x)

        # Non-linear activation
        x = F.relu(x)

        # Regularization
        x = self.dropout(x)

        # Output projection (logits)
        x = self.fc2(x)

        return x
