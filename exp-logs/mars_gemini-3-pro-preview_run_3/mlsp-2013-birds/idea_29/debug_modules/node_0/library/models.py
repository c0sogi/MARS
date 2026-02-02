import torch
import torch.nn as nn
import timm
from library.config import Config


def get_cnn_model(model_name, pretrained=True, num_classes=Config.NUM_CLASSES):
    """
    Factory function to create CNN models using timm.
    Initializes models with pre-trained weights and modifies the head for the specific number of classes.

    Args:
        model_name (str): Name of the model architecture (e.g., 'resnet18').
        pretrained (bool): Whether to load pretrained ImageNet weights. Defaults to True.
        num_classes (int): Number of output classes. Defaults to Config.NUM_CLASSES.

    Returns:
        nn.Module: The constructed PyTorch model.
    """
    if model_name not in Config.CNN_MODELS:
        raise ValueError(
            f"Model {model_name} is not defined in Config.CNN_MODELS: {Config.CNN_MODELS}"
        )

    # Create model using timm
    # in_chans=3 matches the 3-channel replication strategy in data loader
    # timm automatically handles the replacement of the classification head when num_classes is specified
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=Config.IMG_CHANNELS,
    )

    return model


class SymbolicMLP(nn.Module):
    """
    Shallow Multi-Layer Perceptron for processing Bag-of-Audio-Words features.
    Designed for cluster-frequency analysis.

    Architecture: Linear -> ReLU -> Dropout -> Linear
    """

    def __init__(
        self,
        input_dim=Config.MLP_INPUT_DIM,
        hidden_dim=Config.MLP_HIDDEN_DIM,
        dropout=Config.MLP_DROPOUT,
        num_classes=Config.NUM_CLASSES,
    ):
        """
        Args:
            input_dim (int): Input feature dimension (default: 100).
            hidden_dim (int): Hidden layer dimension (default: 128).
            dropout (float): Dropout probability (default: 0.5).
            num_classes (int): Number of output classes (default: 19).
        """
        super(SymbolicMLP, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        return self.network(x)
