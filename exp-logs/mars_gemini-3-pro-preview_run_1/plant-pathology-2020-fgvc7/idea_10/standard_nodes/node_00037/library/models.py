import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


def get_model(pretrained=Config.PRETRAINED):
    """
    Constructs the ResNet34 model for Apple Disease Detection.

    Args:
        pretrained (bool): Whether to load ImageNet pre-trained weights.
                           Defaults to the value in Config.

    Returns:
        torch.nn.Module: The configured model on the specified device.
    """
    # Load the ResNet34 backbone
    # We use the 'pretrained' argument which accepts the boolean from our Config.
    # In modern torchvision, pretrained=True maps to the default ImageNet weights.
    model = models.resnet34(pretrained=pretrained)

    # Modify the classifier head
    # The default ResNet architecture applies Global Average Pooling before this layer.
    # We replace the final Linear layer to match our number of target classes (4).
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    # Move the model to the computation device defined in Config (e.g., CUDA)
    model = model.to(Config.DEVICE)

    return model
