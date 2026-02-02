import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(pretrained=Config.PRETRAINED):
    """
    Creates and returns the ResNet34 model architecture for apple disease classification.

    The model is instantiated using the `timm` library. By specifying `num_classes`,
    the default ImageNet head is replaced with a Global Average Pooling layer
    followed by a single Fully Connected (Linear) layer with 4 outputs,
    as required by the task description.

    Args:
        pretrained (bool): If True, initializes the backbone with weights pre-trained on ImageNet.
                           Defaults to Config.PRETRAINED.

    Returns:
        torch.nn.Module: The ResNet34 model moved to the configured device (GPU/CPU).
    """

    # Instantiate the model
    # Config.MODEL_NAME is 'resnet34'
    # Config.NUM_CLASSES is 4
    model = timm.create_model(
        Config.MODEL_NAME, pretrained=pretrained, num_classes=Config.NUM_CLASSES
    )

    # Move the model to the computation device defined in Config
    model = model.to(Config.DEVICE)

    return model
