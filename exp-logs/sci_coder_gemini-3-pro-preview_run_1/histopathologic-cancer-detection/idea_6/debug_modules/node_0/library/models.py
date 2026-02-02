import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class ModifiedDenseNet(nn.Module):
    """
    A DenseNet architecture adapted for small input images (e.g., 48x48) by modifying the input stem.

    Standard DenseNet Stem: 7x7 Conv (stride 2) -> MaxPool (stride 2) => 4x downsampling.
    Modified Stem: 3x3 Conv (stride 1) -> Identity => 1x downsampling.

    This allows the deep layers to process high-resolution feature maps, which is critical
    for detecting small tumor features in small patches.
    """

    def __init__(self, model_name, pretrained=True, num_classes=1):
        super(ModifiedDenseNet, self).__init__()

        # 1. Load the base model with appropriate weights
        if model_name == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.net = models.densenet121(weights=weights)
        elif model_name == "densenet169":
            weights = models.DenseNet169_Weights.DEFAULT if pretrained else None
            self.net = models.densenet169(weights=weights)
        else:
            raise ValueError(
                f"Model architecture '{model_name}' is not supported. "
                "Choose 'densenet121' or 'densenet169'."
            )

        # 2. Modify the Stem
        # The 'features' attribute in torchvision DenseNet is a Sequential container.
        # It contains: conv0, norm0, relu0, pool0, denseblock1, ...

        # Replace conv0: 7x7 stride 2 -> 3x3 stride 1
        # We keep out_channels=64 to match the input of the first dense block.
        # bias=False is used because BatchNorm follows immediately.
        self.net.features.conv0 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Remove pool0: MaxPool stride 2 -> Identity
        # This prevents the early loss of spatial resolution.
        self.net.features.pool0 = nn.Identity()

        # 3. Modify the Classifier
        # Replace the final fully connected layer to match the target number of classes.
        in_features = self.net.classifier.in_features
        self.net.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass delegating to the modified torchvision model.
        """
        return self.net(x)


def get_model(model_name, pretrained=True):
    """
    Factory function to create a ModifiedDenseNet instance.

    Args:
        model_name (str): Name of the architecture ('densenet121' or 'densenet169').
        pretrained (bool): Whether to initialize with ImageNet weights.

    Returns:
        nn.Module: The configured model.
    """
    return ModifiedDenseNet(
        model_name=model_name, pretrained=pretrained, num_classes=Config.NUM_CLASSES
    )
