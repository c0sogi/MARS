import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


def get_model(model_name: str, config: Config, pretrained: bool = True) -> nn.Module:
    """
    Factory function to initialize and return the specified model architecture.

    Args:
        model_name (str): Name of the architecture ('resnet18', 'efficientnet_b0', 'densenet121').
        config (Config): Configuration object containing model hyperparameters (NUM_CLASSES, DEVICE).
        pretrained (bool): Whether to load ImageNet pre-trained weights.

    Returns:
        nn.Module: The initialized PyTorch model moved to the configured device.
    """

    # Select weights
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)

        # Modify the fully connected layer
        # ResNet18 fc: Linear(in_features=512, out_features=1000)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        # Modify the classifier block
        # EfficientNet classifier is usually a Sequential(Dropout, Linear)
        # We preserve the dropout and replace the linear layer
        dropout_p = model.classifier[0].p
        num_ftrs = model.classifier[1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(num_ftrs, config.NUM_CLASSES),
        )

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)

        # Modify the classifier layer
        # DenseNet classifier: Linear(in_features=1024, out_features=1000)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, config.NUM_CLASSES)

    else:
        raise ValueError(
            f"Architecture '{model_name}' is not supported. "
            f"Choose from {config.ARCHITECTURES}"
        )

    # Move model to the specified device (GPU/CPU)
    model = model.to(config.DEVICE)

    return model
