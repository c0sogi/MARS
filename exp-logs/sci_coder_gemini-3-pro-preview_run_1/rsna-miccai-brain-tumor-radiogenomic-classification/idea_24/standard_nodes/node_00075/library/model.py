import torch
import torch.nn as nn
import timm
from library.config import Config


class CAWIVModel(nn.Module):
    """
    Centroid-Aligned Volumetric Network.
    Simplified to use standard 3-channel input to avoid optimization instability.
    Cite solution_lesson_node_00009: Naive Channel-Stacking Destabilizes 2D CNNs.
    Cite solution_lesson_node_00025: Avoid learnable input projections/modifications.
    """

    def __init__(self, model_name=Config.BACKBONE, pretrained=True):
        super().__init__()

        # Load backbone with 1 output class for binary classification
        # Standard EfficientNet expects 3 channels, which matches our new input.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=1
        )

        # Add Dropout to the classifier for regularization
        self._modify_classifier()

    def _modify_classifier(self):
        """
        Injects Dropout into the classifier head.
        """
        if hasattr(self.backbone, "classifier"):
            # EfficientNet classifier is typically a Linear layer
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=Config.DROPOUT_RATE), nn.Linear(in_features, 1)
            )
        elif hasattr(self.backbone, "fc"):
            # ResNet style
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=Config.DROPOUT_RATE), nn.Linear(in_features, 1)
            )
        elif hasattr(self.backbone, "head"):
            # ViT style
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.Dropout(p=Config.DROPOUT_RATE), nn.Linear(in_features, 1)
            )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 9, H, W).
        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        return self.backbone(x)
