import torch
import torch.nn as nn
import timm
from library.config import (
    BACKBONE,
    NUM_CHANNELS,
    DROPOUT_RATE,
    NUM_CLASSES,
)


class SICAVModel(nn.Module):
    """
    Standard EfficientNet-B0 Model.

    Reverted to standard 3-channel input (FLAIR, T1wCE, T2w) based on lessons learned.
    Cite solution_lesson_node_00002: Middle slice baseline.
    """

    def __init__(
        self,
        backbone_name=BACKBONE,
        num_channels=NUM_CHANNELS,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        pretrained=True,
    ):
        super(SICAVModel, self).__init__()

        # 1. Load Backbone
        # num_classes=0 removes the default classifier head, returning pooled features
        # in_chans=3 allows loading standard pretrained weights without modification
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=num_channels,
            drop_rate=dropout_rate,  # Explicitly pass dropout rate to backbone if supported
        )

        # 2. Define Classifier Head
        # Get the number of features output by the backbone
        num_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 9, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
