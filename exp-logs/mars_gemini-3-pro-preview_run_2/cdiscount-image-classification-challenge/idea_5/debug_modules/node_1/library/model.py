import torch.nn as nn
from torchvision import models
from library.config import Config


def get_model(pretrained=True):
    """
    Initializes the ResNet-34 model for product categorization.

    This function loads a ResNet-34 backbone, optionally with pre-trained ImageNet weights,
    and replaces the final fully connected layer to match the number of classes in the
    Cdiscount dataset.

    Args:
        pretrained (bool): If True, loads the model with 'DEFAULT' (ImageNet) weights.
                           If False, initializes random weights. Defaults to True.

    Returns:
        nn.Module: The modified ResNet-34 model ready for training or inference.
    """
    # Determine weights to load
    if pretrained:
        weights = models.ResNet34_Weights.DEFAULT
    else:
        weights = None

    # Load the ResNet-34 architecture
    model = models.resnet34(weights=weights)

    # Replace the final fully connected layer (Head)
    # ResNet-34 uses a 512-dimensional feature vector before the classification head.
    # We replace the original 1000-class layer with one outputting Config.NUM_CLASSES.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    return model
