import torch
import torch.nn as nn
import timm
from library.config import Config


class SpeciesModel(nn.Module):
    """
    Model for species classification using timm.
    Defaults to EfficientNet-B0 as per Config.MODEL_NAME.
    """

    def __init__(self):
        super(SpeciesModel, self).__init__()

        # Load the pre-trained model using timm
        # timm handles the classifier head replacement automatically via num_classes
        self.model = timm.create_model(
            Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=Config.NUM_CLASSES,
            drop_rate=Config.DROPOUT_RATE,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images (N, 3, H, W).

        Returns:
            torch.Tensor: Raw logits for each class (N, NUM_CLASSES).
        """
        return self.model(x)
