import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


def get_model(
    model_name: str, num_classes: int = Config.NUM_CLASSES, pretrained: bool = True
) -> nn.Module:
    """
    Factory function to instantiate models for the Heterogeneous Ensemble.

    Supported Architectures:
    - 'resnet18': Balances capacity and generalization.
    - 'efficientnet_b0': Parameter efficient, uses MBConv.
    - 'densenet121': Feature reuse, robust for small data.

    Args:
        model_name (str): Name of the architecture.
        num_classes (int): Number of output classes (default: 19).
        pretrained (bool): Whether to load ImageNet pretrained weights (default: True).

    Returns:
        nn.Module: The modified PyTorch model ready for training.
    """
    model_name = model_name.lower()

    if model_name == "resnet18":
        # Load ResNet-18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)

        # Replace the final fully connected layer
        # ResNet structure: model.fc
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        # Load EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        # Replace the final classification layer
        # EfficientNet structure: model.classifier is Sequential(Dropout, Linear)
        # We target the Linear layer at index 1
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "densenet121":
        # Load DenseNet-121
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)

        # Replace the final classifier
        # DenseNet structure: model.classifier is a Linear layer
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Model '{model_name}' not supported. " f"Choose from: {Config.MODELS}"
        )

    return model
