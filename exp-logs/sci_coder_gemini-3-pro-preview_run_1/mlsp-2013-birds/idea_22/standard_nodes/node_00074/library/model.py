import torch
import torch.nn as nn
from torchvision import models
from library import config, utils


def get_model(device=config.DEVICE, weights_path=None):
    """
    Initializes the ResNet-34 model, modifies the head for the specific number of classes,
    and optionally loads pretrained weights.

    Args:
        device (str): Device to move the model to ('cpu' or 'cuda').
        weights_path (str, optional): Path to a checkpoint file to load weights from.

    Returns:
        torch.nn.Module: The configured ResNet-34 model.
    """
    # Initialize ResNet-34 with ImageNet weights
    # We use the modern weights API to ensure we get the V1 weights
    weights = models.ResNet34_Weights.IMAGENET1K_V1
    model = models.resnet34(weights=weights)

    # Replace the final fully connected layer
    # The original fc layer is Linear(in_features=512, out_features=1000, bias=True)
    # We replace it with a Linear layer projecting to the number of bird species (19)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, config.NUM_CLASSES)

    # Load weights if a path is provided
    if weights_path:
        # Use the utility function from library/utils.py
        # This handles loading the state_dict, mapping to device, and handling 'module.' prefixes
        try:
            utils.load_checkpoint(weights_path, model, device=device)
        except Exception as e:
            # Re-raise exception after logging to ensure the failure is noticed
            raise RuntimeError(
                f"Failed to load weights from {weights_path}. Error: {e}"
            )

    # Move model to the specified device
    model.to(device)

    return model
