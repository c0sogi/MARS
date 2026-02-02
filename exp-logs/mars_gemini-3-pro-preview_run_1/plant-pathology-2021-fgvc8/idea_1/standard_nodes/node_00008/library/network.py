import torch
import torch.nn as nn
import timm
from library.config import Config


class AppleDiseaseModel(nn.Module):
    """
    Apple Disease Detection Model based on EfficientNetV2-Small.

    Architecture:
    - Backbone: Fused-MBConv (EfficientNetV2-Small) pre-trained on ImageNet-1k.
    - Pooling: Global Average Pooling (part of the backbone when num_classes=0).
    - Head: Single Fully Connected (Linear) layer mapping features to class logits.
    """

    def __init__(self, model_name=Config.MODEL_NAME, pretrained=True):
        """
        Initializes the model.

        Args:
            model_name (str): The name of the timm model to load. Defaults to Config.MODEL_NAME.
            pretrained (bool): Whether to load pre-trained weights. Defaults to True.
        """
        super(AppleDiseaseModel, self).__init__()

        # Load the backbone model using timm.
        # Setting num_classes=0 removes the default classifier but keeps the
        # Global Average Pooling (GAP) layer, returning the pooled feature vector.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        # Retrieve the number of features output by the backbone
        in_features = self.backbone.num_features

        # Define the custom classification head
        # Maps backbone features to the number of target classes (6)
        self.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Raw logits of shape (B, NUM_CLASSES).
        """
        # Pass input through the backbone to get pooled features
        features = self.backbone(x)

        # Pass features through the classification head to get logits
        logits = self.fc(features)

        return logits
