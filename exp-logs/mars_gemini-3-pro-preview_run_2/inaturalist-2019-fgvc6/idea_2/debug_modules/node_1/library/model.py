import torch.nn as nn
from torchvision import models
from library.config import Config


def create_model(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
    """
    Creates an EfficientNet-B4 model customized for the iNaturalist dataset.

    Args:
        num_classes (int): The number of target classes (1010).
        pretrained (bool): Whether to initialize with ImageNet weights.

    Returns:
        nn.Module: The PyTorch model.
    """
    # Select weights based on pretrained flag
    if pretrained:
        weights = models.EfficientNet_B4_Weights.DEFAULT
    else:
        weights = None

    # Instantiate the base EfficientNet-B4 model
    model = models.efficientnet_b4(weights=weights)

    # The classifier in torchvision's EfficientNet is a Sequential module:
    # (0): Dropout
    # (1): Linear
    # We need to replace the Linear layer (index 1) to match our number of classes.

    # Retrieve the number of input features for the final layer (1792 for B4)
    in_features = model.classifier[1].in_features

    # Replace the fully connected layer
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
