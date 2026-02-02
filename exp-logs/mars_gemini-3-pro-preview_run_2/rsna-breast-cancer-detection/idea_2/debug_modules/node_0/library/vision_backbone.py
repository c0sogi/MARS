import timm
import torch
import torch.nn as nn
from library.config import BACKBONE_NAME, DEVICE


def get_image_encoder(name=BACKBONE_NAME, pretrained=True):
    """
    Initializes the deep learning feature extractor (backbone).

    Uses timm to load a ResNet18 model, removes the classification head
    to output raw embeddings (size 512), and freezes all parameters.

    Args:
        name (str): The name of the model architecture to load (default: 'resnet18').
        pretrained (bool): Whether to load pre-trained ImageNet weights (default: True).

    Returns:
        torch.nn.Module: The frozen feature extractor model on the configured device.
    """
    # Create the model using timm
    # num_classes=0 removes the final linear layer and returns the pooled features
    model = timm.create_model(name, pretrained=pretrained, num_classes=0)

    # Freeze all parameters to prevent gradient updates
    for param in model.parameters():
        param.requires_grad = False

    # Move the model to the appropriate device (GPU/CPU) defined in config
    model = model.to(DEVICE)

    # Set the model to evaluation mode
    # This is crucial for layers like BatchNorm to work correctly during inference/extraction
    model.eval()

    return model
