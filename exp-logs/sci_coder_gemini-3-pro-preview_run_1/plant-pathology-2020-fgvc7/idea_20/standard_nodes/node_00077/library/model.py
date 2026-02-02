import torch
import torch.nn as nn
import torchvision
from library.config import Config


def get_model(pretrained=Config.PRETRAINED, n_classes=Config.N_CLASSES):
    """
    Initializes a ResNet34 model for the apple disease detection task.

    The model uses a ResNet34 backbone, optionally initialized with ImageNet weights.
    The final fully connected layer is replaced to map the 512 backbone features
    to the target class probabilities.

    Args:
        pretrained (bool): If True, loads weights pretrained on ImageNet.
                           Defaults to Config.PRETRAINED.
        n_classes (int): The number of target classes.
                         Defaults to Config.N_CLASSES.

    Returns:
        torch.nn.Module: The modified ResNet34 model.
    """
    # Select weights based on the pretrained flag
    # torchvision 0.13+ uses the 'weights' parameter instead of 'pretrained'
    if pretrained:
        weights = torchvision.models.ResNet34_Weights.DEFAULT
    else:
        weights = None

    # Load the ResNet34 backbone
    model = torchvision.models.resnet34(weights=weights)

    # Modify the classifier head
    # Standard ResNet34 structure:
    # ...
    # (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    # (fc): Linear(in_features=512, out_features=1000, bias=True)

    # We replace the 'fc' layer to output n_classes instead of 1000.
    # The Global Average Pooling (avgpool) is preserved automatically.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)

    return model
