import torch
import torch.nn as nn
import torchvision.models as models
from library.config import NUM_CLASSES, PRETRAINED


def get_model(pretrained=PRETRAINED):
    """
    Constructs a ResNet-18 model customized for the Cdiscount product categorization task.

    The model uses a ResNet-18 backbone. The final fully connected layer is replaced
    to output logits for the specific number of classes in the dataset (5270).

    Args:
        pretrained (bool): If True, loads weights pre-trained on ImageNet.
                           Defaults to the value in library.config.PRETRAINED.

    Returns:
        nn.Module: The modified ResNet-18 model.
    """
    # Determine weights based on pretrained flag
    # Using the modern torchvision weights API
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    # Instantiate the ResNet-18 model
    model = models.resnet18(weights=weights)

    # Modify the final fully connected layer
    # ResNet-18's fc layer has 512 input features
    in_features = model.fc.in_features

    # Replace with a new Linear layer mapping to NUM_CLASSES
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    # Initialize the weights of the new layer
    # Xavier Uniform is generally a good default for linear layers
    nn.init.xavier_uniform_(model.fc.weight)

    # Initialize bias to zero if it exists
    if model.fc.bias is not None:
        nn.init.constant_(model.fc.bias, 0)

    return model
