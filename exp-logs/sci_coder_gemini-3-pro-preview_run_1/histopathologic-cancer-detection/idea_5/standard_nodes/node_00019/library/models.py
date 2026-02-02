import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class ModifiedDenseNet(nn.Module):
    """
    DenseNet121 architecture adapted for small patch classification.

    Modifications:
    1. Stem: Replaces the standard 7x7 stride-2 conv and 3x3 stride-2 pool
       with a single 3x3 stride-1 conv to preserve spatial resolution.
    2. Head: Replaces the classifier with a binary output layer.
    """

    def __init__(self, pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES):
        super(ModifiedDenseNet, self).__init__()

        # Load base model with optional pretrained weights
        weights = "IMAGENET1K_V1" if pretrained else None
        self.base_model = models.densenet121(weights=weights)

        if Config.MODIFY_STEM:
            # Replace the first convolution (conv0)
            # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Modified: Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.base_model.features.conv0 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )

            # Remove the first pooling layer (pool0)
            # Original: MaxPool2d(kernel_size=3, stride=2, padding=1)
            # Modified: Identity (pass-through)
            self.base_model.features.pool0 = nn.Identity()

        # Replace the classifier head
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


class ModifiedResNet(nn.Module):
    """
    ResNet50 architecture adapted for small patch classification.

    Modifications:
    1. Stem: Replaces the standard 7x7 stride-2 conv and 3x3 stride-2 pool
       with a single 3x3 stride-1 conv to preserve spatial resolution.
    2. Head: Replaces the fully connected layer with a binary output layer.
    """

    def __init__(self, pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES):
        super(ModifiedResNet, self).__init__()

        # Load base model with optional pretrained weights
        weights = "IMAGENET1K_V1" if pretrained else None
        self.base_model = models.resnet50(weights=weights)

        if Config.MODIFY_STEM:
            # Replace the first convolution (conv1)
            # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Modified: Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.base_model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )

            # Remove the first pooling layer (maxpool)
            # Original: MaxPool2d(kernel_size=3, stride=2, padding=1)
            # Modified: Identity (pass-through)
            self.base_model.maxpool = nn.Identity()

        # Replace the classifier head (fc)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


def get_model(model_name, pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES):
    """
    Factory function to instantiate the requested model architecture.

    Args:
        model_name (str): Name of the model ('densenet121' or 'resnet50').
        pretrained (bool): Whether to load ImageNet weights.
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: The instantiated PyTorch model.
    """
    if model_name == "densenet121":
        return ModifiedDenseNet(pretrained=pretrained, num_classes=num_classes)
    elif model_name == "resnet50":
        return ModifiedResNet(pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(
            f"Model '{model_name}' is not supported. Available models: {Config.MODEL_NAMES}"
        )
