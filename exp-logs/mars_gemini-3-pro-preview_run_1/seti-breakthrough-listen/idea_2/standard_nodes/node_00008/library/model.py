import torch
import torch.nn as nn
from torchvision import models
from library.config import IN_CHANNELS, NUM_CLASSES


def get_multichannel_resnet(pretrained=True):
    """
    Constructs a ResNet-18 model modified for 6-channel input and binary classification.

    The first convolutional layer is adapted to accept the 6-channel cadence snippet
    (ABACAD panels). The final fully connected layer is adapted to output a single logit.

    Args:
        pretrained (bool): If True, initializes the backbone with ImageNet weights.

    Returns:
        model (torch.nn.Module): The modified ResNet-18 model.
    """
    # Load the base ResNet-18 model
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.resnet18(weights=weights)

    # --- Modify First Convolutional Layer ---
    # Original: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # We need to change in_channels from 3 to IN_CHANNELS (6)
    original_conv1 = model.conv1

    model.conv1 = nn.Conv2d(
        in_channels=IN_CHANNELS,
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias,
    )

    # Initialize the new conv1 weights
    # If pretrained, we average the original RGB weights and replicate them for the 6 channels.
    # This helps preserve low-level feature detectors (edges, textures) from ImageNet.
    if pretrained and weights is not None:
        with torch.no_grad():
            # shape: [out_channels, 3, k, k]
            original_weights = original_conv1.weight

            # Average across the RGB dimension -> shape: [out_channels, 1, k, k]
            avg_weights = torch.mean(original_weights, dim=1, keepdim=True)

            # Replicate across the new input channels -> shape: [out_channels, 6, k, k]
            new_weights = avg_weights.repeat(1, IN_CHANNELS, 1, 1)

            # Copy to the new layer
            model.conv1.weight.copy_(new_weights)

    # --- Modify Final Fully Connected Layer ---
    # Original: nn.Linear(512, 1000)
    # We need to change out_features to NUM_CLASSES (1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    return model
