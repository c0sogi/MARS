import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdCNN(nn.Module):
    """
    Convolutional Neural Network for Bird Species Classification.

    Wraps timm models to provide a consistent interface for the ensemble.
    Supports ResNet18, EfficientNet-B0, and DenseNet121 backbones.
    """

    def __init__(self, backbone_name, pretrained=Config.PRETRAINED):
        """
        Initialize the CNN model.

        Args:
            backbone_name (str): Name of the backbone architecture (e.g., 'resnet18').
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(BirdCNN, self).__init__()

        if backbone_name not in Config.CNN_MODELS:
            raise ValueError(
                f"Backbone '{backbone_name}' is not supported. "
                f"Available options: {Config.CNN_MODELS}"
            )

        # Create the model using timm
        # num_classes ensures the final layer is adapted to our specific problem
        # in_chans=3 matches the 3-channel replicated spectrograms
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=Config.NUM_CLASSES,
            in_chans=3,
        )

    def forward(self, x):
        """
        Forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        return self.backbone(x)


class BirdMLP(nn.Module):
    """
    Multi-Layer Perceptron for processing Bag-of-Audio-Words features.

    Implements a lightweight architecture for the shallow learning stream:
    Input -> [Linear -> ReLU -> Dropout] x N -> Linear -> Output
    """

    def __init__(self):
        super(BirdMLP, self).__init__()

        layers = []
        input_dim = Config.MLP_INPUT_DIM

        # Dynamically build hidden layers based on config
        for hidden_dim in Config.MLP_HIDDEN_DIMS:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(Config.MLP_DROPOUT))
            input_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(input_dim, Config.NUM_CLASSES))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input feature vectors of shape (Batch, Input_Dim).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        return self.model(x)
