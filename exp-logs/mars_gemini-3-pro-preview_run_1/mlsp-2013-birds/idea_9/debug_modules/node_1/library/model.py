import torch
import torch.nn as nn
import torchvision.models as models
from library.config import NUM_CLASSES


def get_bird_model(pretrained=True):
    """
    Initializes and returns a ResNet-34 model for bird species classification.

    The model expects 3-channel input (RGB) which is handled by the data pipeline's
    channel replication strategy. It outputs raw logits for the 19 species classes.

    Args:
        pretrained (bool): If True, loads ImageNet pre-trained weights.
                           If False, initializes with random weights.

    Returns:
        torch.nn.Module: The modified ResNet-34 model.
    """
    # Use the modern torchvision weights API if available, otherwise fallback logic could be added
    # but torchvision 0.23+ supports this standardly.
    if pretrained:
        weights = models.ResNet34_Weights.DEFAULT
    else:
        weights = None

    # Load the backbone
    model = models.resnet34(weights=weights)

    # The input layer of ResNet-34 expects 3 channels by default.
    # Our data pipeline replicates the mono spectrogram to 3 channels,
    # so we do not need to modify conv1.

    # Replace the final fully connected layer
    # ResNet-34's fc layer input features is typically 512
    in_features = model.fc.in_features

    # Create a simple linear projection to the number of classes
    # We do not include Sigmoid here because the loss function (BCEWithLogitsLoss)
    # includes it for numerical stability.
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    return model
