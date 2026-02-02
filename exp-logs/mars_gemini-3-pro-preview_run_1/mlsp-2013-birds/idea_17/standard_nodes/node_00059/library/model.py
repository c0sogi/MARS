import torch.nn as nn
from torchvision import models
from library.config import Config


def get_bird_model(pretrained=True):
    """
    Constructs the ResNet-34 model with a custom classification head for bird species prediction.

    Adheres to the architectural constraints:
    - Backbone: Vanilla ResNet-34 (No SE blocks).
    - Head: Dropout (p=0.5) -> Linear.

    Args:
        pretrained (bool): If True, initializes the backbone with ImageNet weights.
                           Defaults to True as per Config.PRETRAINED.

    Returns:
        model (nn.Module): The PyTorch model ready for training or inference.
    """
    # Initialize ResNet-34 backbone
    # Using 'DEFAULT' weights loads the most up-to-date ImageNet weights available in torchvision
    if pretrained:
        weights = models.ResNet34_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet34(weights=weights)

    # The input data from BirdDataset is (Batch, 3, Height, Width),
    # which matches ResNet's expected input, so conv1 remains unchanged.

    # Modify the classification head
    # ResNet-34's final layer is named 'fc' and has 512 input features.
    in_features = model.fc.in_features

    # Replace the standard Linear layer with the required Dropout -> Linear sequence
    # This introduces structural noise for the Noisy Student pipeline
    model.fc = nn.Sequential(
        nn.Dropout(p=Config.DROPOUT_P), nn.Linear(in_features, Config.NUM_CLASSES)
    )

    return model
