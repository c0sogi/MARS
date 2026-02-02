import torch.nn as nn
from torchvision import models
from library.config import Config


def create_model(arch_name, num_classes=Config.NUM_CLASSES, pretrained=True):
    """
    Creates a model instance based on the specified architecture name.
    Supports 'resnet34' and 'densenet121' as per the Heterogeneous Ensemble strategy.

    Args:
        arch_name (str): Name of the architecture ('resnet34' or 'densenet121').
        num_classes (int): Number of output classes. Defaults to Config.NUM_CLASSES.
        pretrained (bool): Whether to load ImageNet pretrained weights. Defaults to True.

    Returns:
        torch.nn.Module: The modified model with a linear classification head.
    """
    # Determine weights
    weights = "IMAGENET1K_V1" if pretrained else None

    if arch_name == "resnet34":
        # Instantiate ResNet-34
        model = models.resnet34(weights=weights)

        # Replace the final fully connected layer
        # ResNet uses 'fc' as the final layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif arch_name == "densenet121":
        # Instantiate DenseNet-121
        # Chosen for architectural diversity and parameter efficiency
        model = models.densenet121(weights=weights)

        # Replace the final classifier layer
        # DenseNet uses 'classifier' as the final layer
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unsupported architecture: {arch_name}. "
            f"Allowed: {Config.TEACHER_ARCHS}"
        )

    return model
