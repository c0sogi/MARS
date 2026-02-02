import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class AppleResNet34(nn.Module):
    """
    ResNet34 architecture for Apple Disease Detection.

    This class initializes a ResNet34 backbone with ImageNet weights and replaces
    the default fully connected head with a new linear layer for the specific
    number of target classes. It relies on the backbone's built-in Global Average
    Pooling (AdaptiveAvgPool2d) to handle variable input sizes (e.g., for
    progressive resizing).
    """

    def __init__(self, num_classes=None, pretrained=True):
        """
        Args:
            num_classes (int, optional): Number of target classes. Defaults to Config.NUM_CLASSES.
            pretrained (bool, optional): If True, initializes with ImageNet weights. Defaults to True.
        """
        super(AppleResNet34, self).__init__()

        if num_classes is None:
            num_classes = Config.NUM_CLASSES

        # Initialize ResNet34 backbone
        # Attempt to use the newer 'weights' API, fallback to 'pretrained' for compatibility
        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet34(weights=weights)
        except (AttributeError, ImportError):
            self.model = models.resnet34(pretrained=pretrained)

        # Replace the classification head
        # The original 'fc' is a Linear layer. We replace it to output the correct number of classes.
        # The preceding layer in ResNet is AdaptiveAvgPool2d((1,1)), which serves as Global Average Pooling.
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input batch of images (B, C, H, W).

        Returns:
            torch.Tensor: Raw logits (B, num_classes).
        """
        return self.model(x)
