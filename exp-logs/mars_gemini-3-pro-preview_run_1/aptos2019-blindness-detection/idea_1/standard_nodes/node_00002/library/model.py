import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class OrdinalEfficientNet(nn.Module):
    """
    EfficientNet-B0 architecture adapted for Ordinal Regression.

    This model replaces the final classification layer of EfficientNet-B0
    with a layer outputting 'NUM_ORDINAL_UNITS' logits. These logits represent
    the ordinal thresholds (P(y > k)) and are intended to be used with
    BCEWithLogitsLoss during training and passed through a Sigmoid during inference.
    """

    def __init__(self, pretrained: bool = Config.PRETRAINED):
        """
        Initializes the model.

        Args:
            pretrained (bool): If True, loads weights pretrained on ImageNet.
                               Defaults to Config.PRETRAINED.
        """
        super(OrdinalEfficientNet, self).__init__()

        # Select weights
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None

        # Load the base model
        self.backbone = efficientnet_b0(weights=weights)

        # Modify the classifier head
        # The classifier in EfficientNet is a nn.Sequential block.
        # Structure: Dropout -> Linear
        classifier = self.backbone.classifier

        # Access the last layer to get input features
        last_layer_idx = len(classifier) - 1
        last_layer = classifier[last_layer_idx]

        if isinstance(last_layer, nn.Linear):
            in_features = last_layer.in_features

            # Replace with new Linear layer for Ordinal Regression
            # Output size is NUM_ORDINAL_UNITS (4 for 5 classes)
            classifier[last_layer_idx] = nn.Linear(
                in_features, Config.NUM_ORDINAL_UNITS
            )
        else:
            # Fallback
            classifier[last_layer_idx] = nn.Linear(1280, Config.NUM_ORDINAL_UNITS)

        self.backbone.classifier = classifier

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, NUM_ORDINAL_UNITS).
        """
        return self.backbone(x)
