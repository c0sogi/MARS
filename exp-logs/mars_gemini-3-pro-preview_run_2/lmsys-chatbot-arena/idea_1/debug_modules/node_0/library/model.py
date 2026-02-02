import torch
import torch.nn as nn
from library.config import Config


class ChatbotMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for Chatbot Preference Prediction.

    This model is designed to work with pre-computed embeddings (Frozen Siamese Transformer approach).
    It takes a dense feature vector as input—constructed from the prompt embedding,
    response embeddings, and their interactions (difference, product)—and predicts
    the logits for the three target classes: Winner Model A, Winner Model B, and Tie.

    Architecture:
        - Input Layer: Matches the combined embedding dimension.
        - Hidden Layers: A sequence of (Linear -> BatchNorm -> ReLU -> Dropout) blocks.
        - Output Layer: Linear layer projecting to the number of classes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list = None,
        dropout_rate: float = None,
        num_classes: int = None,
    ):
        """
        Initialize the ChatbotMLP model.

        Args:
            input_dim (int): The size of the input feature vector.
            hidden_layers (list, optional): List of integers defining the size of each hidden layer.
                                            Defaults to Config.HIDDEN_LAYERS.
            dropout_rate (float, optional): The dropout probability. Defaults to Config.DROPOUT_RATE.
            num_classes (int, optional): The number of output classes. Defaults to Config.NUM_CLASSES.
        """
        super(ChatbotMLP, self).__init__()

        # Set defaults from Config if arguments are not provided
        if hidden_layers is None:
            hidden_layers = Config.HIDDEN_LAYERS
        if dropout_rate is None:
            dropout_rate = Config.DROPOUT_RATE
        if num_classes is None:
            num_classes = Config.NUM_CLASSES

        layers = []
        current_dim = input_dim

        # Build hidden layers
        for h_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        # Output layer
        layers.append(nn.Linear(current_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input feature tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        return self.network(x)
