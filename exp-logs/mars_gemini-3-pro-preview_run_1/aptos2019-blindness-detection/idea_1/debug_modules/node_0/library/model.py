import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from library.config import Config


class OrdinalMobileNetV3(nn.Module):
    """
    MobileNetV3-Small architecture adapted for Ordinal Regression.

    This model replaces the final classification layer of MobileNetV3-Small
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
        super(OrdinalMobileNetV3, self).__init__()

        # Select weights
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None

        # Load the base model
        self.backbone = mobilenet_v3_small(weights=weights)

        # Modify the classifier head
        # The classifier in MobileNetV3 is a nn.Sequential block.
        # Structure: Linear -> Hardswish -> Dropout -> Linear
        # We want to replace the final Linear layer.
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
            # Fallback logic if architecture changes in future torchvision versions
            # Assuming the standard structure, the penultimate output is 1024
            classifier[last_layer_idx] = nn.Linear(1024, Config.NUM_ORDINAL_UNITS)

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
