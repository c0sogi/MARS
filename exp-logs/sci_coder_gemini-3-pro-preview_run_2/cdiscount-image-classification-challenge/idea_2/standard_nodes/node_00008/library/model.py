import torch
import torch.nn as nn
from torchvision import models
from library import config


def get_model():
    """
    Instantiates the ResNet-50 model architecture, modifies the final classification head
    to match the dataset's number of classes, and moves the model to the configured device.

    Returns:
        model (torch.nn.Module): The configured ResNet-50 model.
    """
    print(f"Creating model: {config.MODEL_ARCH} (Pretrained={config.PRETRAINED})")

    # Determine weights based on config
    if config.PRETRAINED:
        weights = models.ResNet50_Weights.DEFAULT
    else:
        weights = None

    # Load the backbone
    model = models.resnet50(weights=weights)

    # Modify the final fully connected layer
    # ResNet-50 has an 'fc' layer with 2048 input features
    num_ftrs = model.fc.in_features

    # Construct the new head
    # We check config.DROPOUT to see if regularization is needed in the head
    if hasattr(config, "DROPOUT") and config.DROPOUT > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=config.DROPOUT), nn.Linear(num_ftrs, config.NUM_CLASSES)
        )
    else:
        model.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)

    # Move the model to the computation device (GPU/CPU)
    model = model.to(config.DEVICE)

    return model
