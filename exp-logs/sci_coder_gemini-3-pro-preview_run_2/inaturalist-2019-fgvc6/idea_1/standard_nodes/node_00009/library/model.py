import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class EfficientNetB0Baseline(nn.Module):
    """
    EfficientNet-B0 based model for species classification.
    Replaces the standard ImageNet head with a simplified linear head
    mapping backbone features directly to the target classes.
    """

    def __init__(self):
        super(EfficientNetB0Baseline, self).__init__()

        # Load the pre-trained EfficientNet-B0 model
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if Config.PRETRAINED else None
        self.model = efficientnet_b0(weights=weights)

        # EfficientNet-B0 backbone outputs 1280 channels after the pooling layer
        # The default classifier is: Dropout -> Linear(1280, 1000)
        # We replace this with: Dropout -> Linear(1280, NUM_CLASSES)
        in_features = 1280

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(in_features, Config.NUM_CLASSES),
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
