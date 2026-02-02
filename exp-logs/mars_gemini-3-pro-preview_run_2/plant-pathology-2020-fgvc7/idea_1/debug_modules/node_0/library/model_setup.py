import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


def build_model(pretrained=True, num_classes=Config.NUM_CLASSES):
    """
    Constructs a ResNet-18 model for Apple Disease Detection.

    Initializes a ResNet-18 backbone. If pretrained is True, loads weights
    pre-trained on ImageNet. The final fully connected layer is replaced
    to output logits for the specific number of target classes.

    Args:
        pretrained (bool): If True, loads weights pre-trained on ImageNet.
                           Defaults to True.
        num_classes (int): Number of output classes. Defaults to Config.NUM_CLASSES.

    Returns:
        torch.nn.Module: The modified ResNet-18 model ready for training.
    """
    # Determine weights to load
    if pretrained:
        # models.ResNet18_Weights.DEFAULT corresponds to the best available weights (IMAGENET1K_V1)
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    # Load the ResNet-18 model
    model = models.resnet18(weights=weights)

    # Input features for the final fully connected layer in ResNet-18 is 512
    in_features = model.fc.in_features

    # Replace the existing fully connected layer with a new one
    # This layer maps the 512 feature vector to the 4 class logits
    model.fc = nn.Linear(in_features, num_classes)

    return model
