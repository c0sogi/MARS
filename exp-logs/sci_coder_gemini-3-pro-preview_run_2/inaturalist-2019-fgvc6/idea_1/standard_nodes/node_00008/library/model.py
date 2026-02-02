import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class SpeciesModel(nn.Module):
    """
    EfficientNet-B0 based model for species classification.
    Cite solution_lesson_node_00005: Upgraded to EfficientNet-B0 for better fine-grained performance.
    """

    def __init__(self):
        super(SpeciesModel, self).__init__()

        # Load the pre-trained EfficientNet-B0 model
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if Config.PRETRAINED else None
        self.model = efficientnet_b0(weights=weights)

        # EfficientNet-B0 classifier is: Sequential(Dropout, Linear(1280, 1000))
        # We replace the final Linear layer to match NUM_CLASSES
        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images (N, 3, H, W).

        Returns:
            torch.Tensor: Raw logits for each class (N, NUM_CLASSES).
        """
        return self.model(x)


# Alias for backward compatibility
MobileNetV3Baseline = SpeciesModel
