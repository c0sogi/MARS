import torch
import torch.nn as nn
import timm
from library import config


class AppleDiseaseModel(nn.Module):
    def __init__(
        self,
        model_name=config.MODEL_NAME,
        pretrained=True,
        num_classes=config.NUM_CLASSES,
    ):
        """
        Apple Disease Detection Model based on EfficientNetV2.

        Args:
            model_name (str): Name of the timm model backbone.
            pretrained (bool): Whether to load pretrained weights.
            num_classes (int): Number of output classes.
        """
        super(AppleDiseaseModel, self).__init__()

        # Load the backbone using timm.
        # Setting num_classes=0 removes the default classifier and returns
        # the pooled feature vector (flattened).
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        # Retrieve the number of features output by the backbone
        in_features = self.backbone.num_features

        # Define a custom fully connected layer (head)
        # This maps the backbone features to the number of disease classes.
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of images (B, C, H, W).

        Returns:
            torch.Tensor: Raw logits (B, num_classes) suitable for BCEWithLogitsLoss.
        """
        # Extract features from the backbone
        features = self.backbone(x)

        # Pass features through the custom classification head
        logits = self.fc(features)

        return logits
