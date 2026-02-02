import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class ModelFactory:
    """
    Factory class to instantiate heterogeneous CNN architectures for bird classification.
    Supports ResNet-18, EfficientNet-B0, and DenseNet-121.
    """

    @staticmethod
    def create_model(
        architecture_name, num_classes=Config.NUM_CLASSES, pretrained=True
    ):
        """
        Creates and returns a model based on the specified architecture name.

        Args:
            architecture_name (str): Name of the architecture ('resnet18', 'efficientnet_b0', 'densenet121').
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load ImageNet pre-trained weights.

        Returns:
            torch.nn.Module: The instantiated and modified model.
        """
        architecture_name = architecture_name.lower()

        if architecture_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet18(weights=weights)

            # Replace the final fully connected layer
            # ResNet uses 'fc'
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

        elif architecture_name == "densenet121":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.densenet121(weights=weights)

            # Replace the final classifier layer
            # DenseNet uses 'classifier'
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)

        elif architecture_name == "efficientnet_b0":
            weights = (
                models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            )
            model = models.efficientnet_b0(weights=weights)

            # EfficientNet uses 'classifier', which is a Sequential block.
            # Structure: Sequential(Dropout(p=0.2, inplace=True), Linear(...))
            # We replace the Linear layer (index 1) to match num_classes while keeping Dropout.
            last_layer_index = len(model.classifier) - 1
            in_features = model.classifier[last_layer_index].in_features
            model.classifier[last_layer_index] = nn.Linear(in_features, num_classes)

        else:
            raise ValueError(
                f"Unsupported architecture: {architecture_name}. "
                f"Supported: {Config.ARCHITECTURES}"
            )

        return model
