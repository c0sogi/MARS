import torch
import torch.nn as nn
import torchvision
from library.config import Config


class ISICModel(nn.Module):
    """
    EfficientNet-B0 based model for binary skin lesion classification.

    Architecture:
    - Backbone: EfficientNet-B0 (pre-trained on ImageNet)
    - Head: Dropout -> Linear(1) -> Logits
    """

    def __init__(self, pretrained=True):
        """
        Initialize the model.

        Args:
            pretrained (bool): Whether to load ImageNet weights. Defaults to True.
        """
        super(ISICModel, self).__init__()

        # Load the pre-trained EfficientNet-B0 model
        weights = (
            torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        self.model = torchvision.models.efficientnet_b0(weights=weights)

        # EfficientNet classifier is a Sequential block:
        # (0): Dropout(p=0.2, inplace=True)
        # (1): Linear(in_features=1280, out_features=1000, bias=True)

        # We replace the Linear layer for binary classification
        in_features = self.model.classifier[1].in_features

        # Keep the original Dropout (0.2) and replace Linear
        self.model.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        return self.model(x)
