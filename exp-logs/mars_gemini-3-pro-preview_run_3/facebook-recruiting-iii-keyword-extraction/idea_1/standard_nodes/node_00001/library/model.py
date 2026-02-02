import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import MAX_FEATURES, HIDDEN_DIM, TOP_K_TAGS, DROPOUT


class SparseMLP(nn.Module):
    """
    A Sparse Multi-Layer Perceptron (MLP) for multi-label classification
    on high-dimensional sparse text features (TF-IDF).

    Architecture:
    Input (Sparse/Dense Vector) -> Linear -> ReLU -> Dropout -> Linear -> Logits
    """

    def __init__(
        self,
        input_dim=MAX_FEATURES,
        hidden_dim=HIDDEN_DIM,
        output_dim=TOP_K_TAGS,
        dropout_prob=DROPOUT,
    ):
        """
        Initialize the SparseMLP model.

        Args:
            input_dim (int): Dimension of the input feature vector (Vocabulary size).
                             Defaults to MAX_FEATURES from config.
            hidden_dim (int): Dimension of the hidden layer.
                              Defaults to HIDDEN_DIM from config.
            output_dim (int): Number of output classes (Target tags).
                              Defaults to TOP_K_TAGS from config.
            dropout_prob (float): Probability of an element to be zeroed in the Dropout layer.
                                  Defaults to DROPOUT from config.
        """
        super(SparseMLP, self).__init__()

        # First Dense Layer: Project input to hidden space
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Dropout Layer for regularization
        self.dropout = nn.Dropout(p=dropout_prob)

        # Second Dense Layer: Project hidden representation to output tag space
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_dim).
        """
        # Linear projection
        x = self.fc1(x)

        # Activation function (ReLU)
        x = F.relu(x)

        # Dropout
        x = self.dropout(x)

        # Output projection (Logits)
        # We do not apply Sigmoid here as BCEWithLogitsLoss includes it
        logits = self.fc2(x)

        return logits
