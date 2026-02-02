import torch.nn as nn
from torchvision import models
from library.config import Config


def get_model(pretrained=Config.PRETRAINED):
    """
    Constructs the ResNet-34 model for bird species classification.

    The architecture consists of a ResNet-34 backbone initialized with ImageNet weights (optional),
    followed by a standard Linear classification head projecting to the target number of classes.

    Args:
        pretrained (bool): If True, loads ImageNet pretrained weights (IMAGENET1K_V1).
                           Defaults to Config.PRETRAINED.

    Returns:
        torch.nn.Module: The modified ResNet-34 model ready for training/inference.
    """
    # Select weights based on configuration
    if pretrained:
        weights = models.ResNet34_Weights.DEFAULT
    else:
        weights = None

    # Instantiate the backbone
    model = models.resnet34(weights=weights)

    # Modify the classification head
    # The original ResNet-34 fc layer is: Linear(in_features=512, out_features=1000, bias=True)
    in_features = model.fc.in_features

    # Replace with a new Linear layer for the specific number of bird species
    # Note: No Dropout is added here. The strategy relies on Sharpness-Aware Minimization (SAM)
    # for regularization instead of structural noise like Dropout.
    model.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    return model
