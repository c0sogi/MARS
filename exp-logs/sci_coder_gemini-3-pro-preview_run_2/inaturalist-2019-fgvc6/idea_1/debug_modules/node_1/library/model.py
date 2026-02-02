import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from library.config import Config


class MobileNetV3Baseline(nn.Module):
    """
    MobileNetV3-Large based model for species classification.
    Replaces the standard ImageNet head with a simplified linear head
    mapping backbone features directly to the target classes.
    """

    def __init__(self):
        super(MobileNetV3Baseline, self).__init__()

        # Load the pre-trained MobileNetV3-Large model
        weights = (
            MobileNet_V3_Large_Weights.IMAGENET1K_V1 if Config.PRETRAINED else None
        )
        self.model = mobilenet_v3_large(weights=weights)

        # MobileNetV3-Large backbone outputs 960 channels after the pooling layer
        # The default classifier is: Linear(960, 1280) -> Hardswish -> Dropout -> Linear(1280, 1000)
        # We replace this with a simpler head: Dropout -> Linear(960, NUM_CLASSES)
        in_features = 960

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
