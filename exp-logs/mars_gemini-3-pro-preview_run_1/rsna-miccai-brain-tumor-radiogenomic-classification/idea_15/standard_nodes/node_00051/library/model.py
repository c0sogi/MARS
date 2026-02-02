import torch
import torch.nn as nn
import timm
from library.config import BACKBONE_NAME, DROPOUT_RATE, NUM_CLASSES, IN_CHANNELS


class WITSNetwork(nn.Module):
    """
    Standard EfficientNet-B0 with custom classifier head.
    Reverted to 3-channel input to preserve pretrained priors (Cite Lesson 00025).
    """

    def __init__(self):
        super().__init__()

        # Load the backbone (EfficientNet-B0)
        # num_classes=0 returns the model with the classification head removed
        # (but keeps the global pooling, outputting a feature vector)
        # We use standard 3-channel input (IN_CHANNELS=3) so no weight inflation needed.
        # Explicitly pass in_chans to link model to data config (Cite debug_lesson_8)
        self.backbone = timm.create_model(
            BACKBONE_NAME, pretrained=True, num_classes=0, in_chans=IN_CHANNELS
        )

        # Define the classifier head
        # self.backbone.num_features gives the output dimension of the backbone (e.g., 1280 for B0)
        self.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(self.backbone.num_features, NUM_CLASSES),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 9, H, W)
        Returns:
            torch.Tensor: Logits of shape (Batch, 1)
        """
        # Extract features
        features = self.backbone(x)

        # Classify
        logits = self.classifier(features)

        return logits
