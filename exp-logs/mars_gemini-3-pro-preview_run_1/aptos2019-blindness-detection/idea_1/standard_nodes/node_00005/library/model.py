import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class OrdinalEfficientNet(nn.Module):
    """
    EfficientNet-B0 architecture adapted for Ordinal Regression.

    Replaces the final classification layer with 'NUM_ORDINAL_UNITS' logits
    for rank-consistent ordinal regression.
    """

    def __init__(self, pretrained: bool = Config.PRETRAINED):
        super(OrdinalEfficientNet, self).__init__()

        # Select weights
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None

        # Load the base model
        self.backbone = efficientnet_b0(weights=weights)

        # Modify the classifier head
        # EfficientNet classifier is Sequential(Dropout, Linear)
        # We replace the Linear layer (index 1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, Config.NUM_ORDINAL_UNITS)

    def forward(self, x):
        return self.backbone(x)
