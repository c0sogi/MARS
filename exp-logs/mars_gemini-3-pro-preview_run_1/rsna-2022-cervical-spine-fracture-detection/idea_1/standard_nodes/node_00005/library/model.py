import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class FractureClassifier(nn.Module):
    """
    2D CNN for slice-level fracture detection.
    Uses a ResNet18 backbone by default, modifying the final layer
    to output probabilities for 8 classes (C1-C7 + patient_overall).
    """

    def __init__(self, backbone_name=Config.BACKBONE, pretrained=Config.PRETRAINED):
        super(FractureClassifier, self).__init__()

        self.backbone_name = backbone_name

        # Initialize backbone
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features

            # Replace the fully connected layer
            # Add Dropout to prevent overfitting
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=0.5), nn.Linear(in_features, Config.NUM_CLASSES)
            )
        else:
            # Fallback or extension for other models (e.g., efficientnet)
            # For this task, we strictly follow Config.BACKBONE="resnet18"
            raise NotImplementedError(f"Backbone {backbone_name} is not implemented.")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
                              C=3 for 2.5D stacking.

        Returns:
            torch.Tensor: Probabilities of shape (B, NUM_CLASSES).
        """
        # Pass through backbone (which includes the modified FC layer)
        logits = self.backbone(x)

        # Apply Sigmoid to get probabilities [0, 1]
        probs = torch.sigmoid(logits)

        return probs
