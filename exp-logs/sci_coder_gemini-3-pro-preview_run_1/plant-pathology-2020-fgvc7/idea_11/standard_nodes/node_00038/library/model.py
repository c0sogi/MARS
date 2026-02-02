import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class AppleResNet34(nn.Module):
    """
    AppleResNet34 model implementation for Apple Disease Detection.

    This architecture uses a ResNet34 backbone initialized with ImageNet weights.
    The classification head is simplified to a Global Average Pooling layer (inherent in ResNet)
    followed by a single fully connected layer. This design choice avoids over-regularization
    and underfitting on the small dataset.
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        """
        Initializes the model.

        Args:
            pretrained (bool): Whether to load pre-trained ImageNet weights.
                               Defaults to the value in Config.
        """
        super(AppleResNet34, self).__init__()

        # Load ResNet34 backbone
        # Using 'DEFAULT' weights ensures we get the best available pre-trained weights
        if pretrained:
            weights = models.ResNet34_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.resnet34(weights=weights)

        # Retrieve the number of input features for the fully connected layer
        # For ResNet34, this is typically 512
        in_features = self.backbone.fc.in_features

        # Replace the fully connected layer
        # We use a simple Linear layer without Dropout (Config.DROPOUT_RATE is 0.0)
        # The Global Average Pooling is handled by the 'avgpool' layer preceding 'fc' in the ResNet architecture
        self.backbone.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        return self.backbone(x)
