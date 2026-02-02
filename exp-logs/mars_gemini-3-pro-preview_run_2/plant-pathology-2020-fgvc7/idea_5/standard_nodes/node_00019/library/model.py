import torch
import torch.nn as nn
import timm
from library.config import Config


class AppleDiseaseModel(nn.Module):
    """
    Apple Disease Detection Model.

    Architecture:
        - Backbone: EfficientNet-B5 (Noisy Student weights)
        - Head: Dropout(0.4) -> Linear(2)

    Strategy:
        Multi-Label Decomposition. The model outputs 2 independent logits
        corresponding to the presence of 'Rust' and 'Scab'.
    """

    def __init__(self, model_name: str = Config.MODEL_NAME, pretrained: bool = True):
        """
        Args:
            model_name (str): The name of the timm backbone to load.
            pretrained (bool): Whether to load pretrained weights (ImageNet).
        """
        super(AppleDiseaseModel, self).__init__()

        # Load the backbone model
        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        # Replace the classifier head
        # We need to handle different naming conventions in timm (classifier, fc, head)
        if hasattr(self.backbone, "classifier"):
            # EfficientNet family
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=Config.DROPOUT_RATE),
                nn.Linear(in_features, Config.NUM_CLASSES),
            )
        elif hasattr(self.backbone, "fc"):
            # ResNet family
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=Config.DROPOUT_RATE),
                nn.Linear(in_features, Config.NUM_CLASSES),
            )
        elif hasattr(self.backbone, "head"):
            # Vision Transformers
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.Dropout(p=Config.DROPOUT_RATE),
                nn.Linear(in_features, Config.NUM_CLASSES),
            )
        else:
            raise AttributeError(
                f"Could not identify classifier layer for model {model_name}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, 2).
                          Index 0 -> Rust Logit
                          Index 1 -> Scab Logit
        """
        return self.backbone(x)
