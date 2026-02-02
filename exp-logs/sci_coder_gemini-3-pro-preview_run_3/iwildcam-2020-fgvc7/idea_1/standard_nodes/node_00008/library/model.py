import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


def get_species_classifier(
    num_classes: int = Config.NUM_CLASSES,
    pretrained: bool = Config.PRETRAINED,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Constructs a MobileNetV3-Small model customized for species classification.

    The architecture uses a MobileNetV3-Small backbone. The default classifier head
    is replaced with a Dropout layer followed by a Fully Connected (Linear) layer
    to output class probabilities.

    Args:
        num_classes (int): The number of output categories (species).
                           Defaults to Config.NUM_CLASSES.
        pretrained (bool): If True, uses weights pretrained on ImageNet.
                           Defaults to Config.PRETRAINED.
        freeze_backbone (bool): If True, the weights of the feature extractor (backbone)
                                are frozen and will not be updated during training.
                                Defaults to False.

    Returns:
        nn.Module: The PyTorch model ready for training or inference.
    """

    # Determine weights to load
    if Config.MODEL_NAME == "resnet18":
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None
        model = models.resnet18(weights=weights)

        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # Modify the classifier head (fc layer in ResNet)
        in_features = model.fc.in_features
        # Increased dropout to 0.5 to combat overfitting
        model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, num_classes))

    else:
        # Fallback to MobileNetV3
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        else:
            weights = None

        model = models.mobilenet_v3_small(weights=weights)

        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(in_features, num_classes)
        )

    return model
