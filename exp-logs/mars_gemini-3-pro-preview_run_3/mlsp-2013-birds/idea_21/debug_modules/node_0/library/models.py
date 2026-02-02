import torch
import torch.nn as nn
from torchvision import models


def get_model(model_name, num_classes=19, pretrained=True):
    """
    Factory function to instantiate CNN models for Bird Species Classification.

    Args:
        model_name (str): Name of the architecture ('resnet18', 'densenet121', 'efficientnet_b0').
        num_classes (int): Number of output classes (default: 19).
        pretrained (bool): Whether to load ImageNet pre-trained weights (default: True).

    Returns:
        torch.nn.Module: The modified PyTorch model.
    """

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)

        # Replace the final fully connected layer
        # ResNet18 fc in_features is 512
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)

        # Replace the classifier layer
        # DenseNet121 classifier in_features is 1024
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        # EfficientNet classifier is a Sequential block:
        # (0): Dropout
        # (1): Linear
        # We replace the Linear layer at index 1
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Model {model_name} not supported. Choose from: resnet18, densenet121, efficientnet_b0"
        )

    return model
