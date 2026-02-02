import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(
    model_name: str = Config.model_name,
    pretrained: bool = Config.pretrained,
    num_classes: int = Config.num_classes,
) -> nn.Module:
    """
    Initializes the model architecture with pretrained weights and a simple linear head.

    This implementation uses the 'timm' library to create a ResNet34 model.
    It ensures that:
    1. The backbone is initialized with ImageNet weights (if pretrained=True).
    2. The default head is replaced with a simple nn.Linear layer for the specified number of classes.
    3. Complex pooling layers (like GeM) are avoided in favor of standard Global Average Pooling
       (which is the default behavior for ResNet in timm).

    Args:
        model_name (str): The name of the architecture to use (e.g., 'resnet34').
                          Defaults to Config.model_name.
        pretrained (bool): Whether to load pretrained ImageNet weights.
                           Defaults to Config.pretrained.
        num_classes (int): The number of target classes.
                           Defaults to Config.num_classes.

    Returns:
        nn.Module: The initialized PyTorch model.
    """
    # Create the model using timm
    # passing num_classes automatically resets the classifier to a new Linear layer
    # with the correct number of outputs.
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes
    )

    return model
