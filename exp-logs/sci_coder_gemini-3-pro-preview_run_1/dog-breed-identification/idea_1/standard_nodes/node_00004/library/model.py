import torch
import torch.nn as nn
from torchvision import models
from library.config import DEVICE, NUM_CLASSES, DROPOUT_RATE


def get_model(num_classes=NUM_CLASSES, pretrained=True, device=DEVICE):
    """
    Initializes the ResNet-18 model, replaces the fully connected head with one
    matching the number of classes, and moves the model to the specified device.

    Args:
        num_classes (int): The number of output classes (dog breeds).
        pretrained (bool): If True, loads weights pre-trained on ImageNet.
        device (torch.device): The device (CPU/GPU) to load the model onto.

    Returns:
        torch.nn.Module: The configured ResNet-18 model.
    """
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None

    # Load the backbone
    model = models.resnet18(weights=weights)

    # Replace the final fully connected layer
    # ResNet-18 has a final layer named 'fc' with 512 input features
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE), nn.Linear(in_features, num_classes)
    )

    # Move model to the computation device
    model = model.to(device)

    return model


def freeze_backbone(model):
    """
    Freezes all parameters in the model except for the final classification head (fc).
    This is used for the first phase of training (Head Adaptation).
    """
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Then, unfreeze only the final linear layer
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_layer4_and_head(model):
    """
    Unfreezes the final residual block (layer4) and the classification head (fc),
    while keeping the earlier layers (conv1, layer1, layer2, layer3) frozen.
    This is used for the second phase of training (Fine-Tuning).
    """
    # Ensure everything is frozen first to reset state
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last convolutional block (layer4)
    # In ResNet implementations, 'layer4' is the final block of convolutions
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Unfreeze the final linear layer
    for param in model.fc.parameters():
        param.requires_grad = True
