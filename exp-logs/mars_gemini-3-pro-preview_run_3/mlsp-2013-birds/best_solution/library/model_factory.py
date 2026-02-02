import torch
import torch.nn as nn
import torchvision.models as models
import timm
from library.config import Config


class SymbolicMLP(nn.Module):
    """
    A shallow Multi-Layer Perceptron (MLP) for processing Bag-of-Audio-Words features.
    Designed to be lightweight to prevent overfitting on the small dataset.

    Architecture:
        Linear(input_dim -> hidden_dim)
        ReLU
        Dropout
        Linear(hidden_dim -> output_dim)
    """

    def __init__(
        self,
        input_dim=Config.MLP_INPUT_DIM,
        hidden_dim=Config.MLP_HIDDEN_DIM,
        output_dim=Config.NUM_CLASSES,
        dropout_prob=Config.MLP_DROPOUT,
    ):
        super(SymbolicMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def get_cnn_model(model_name, num_classes=Config.NUM_CLASSES, pretrained=True):
    """
    Factory function to initialize CNN models for the texture analysis stream.
    Supports ResNet-18, DenseNet-121, and EfficientNet-B0.

    Args:
        model_name (str): Name of the architecture ('resnet18', 'densenet121', 'efficientnet_b0').
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load ImageNet pre-trained weights.

    Returns:
        torch.nn.Module: The configured model.
    """
    model_name = model_name.lower()

    if model_name == "resnet18":
        # Load ResNet-18 from torchvision
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

        # Replace the final fully connected layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "densenet121":
        # Load DenseNet-121 from torchvision
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)

        # Replace the classifier
        # DenseNet classifier is a Linear layer
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        # Load EfficientNet-B0 from timm
        # timm handles head replacement via num_classes argument
        model = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=num_classes
        )

    else:
        raise ValueError(
            f"Model architecture '{model_name}' not supported. Choose from: resnet18, densenet121, efficientnet_b0"
        )

    return model
