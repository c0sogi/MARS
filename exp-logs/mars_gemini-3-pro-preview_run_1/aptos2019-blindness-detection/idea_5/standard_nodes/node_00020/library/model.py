import torch
import torch.nn as nn
import timm
from library.config import Config


class RetinopathyModel(nn.Module):
    """
    Model for Diabetic Retinopathy Severity Prediction using ConvNeXt-Tiny.

    Architecture:
    - Backbone: ConvNeXt-Tiny (pre-trained on ImageNet-1k).
    - Pooling: Global Average Pooling (handled by timm).
    - Head: Dropout -> Linear Layer (outputs 4 ordinal logits).
    """

    def __init__(self, pretrained: bool = True):
        """
        Initialize the model.

        Args:
            pretrained (bool): Whether to load pre-trained ImageNet weights.
        """
        super(RetinopathyModel, self).__init__()

        # Load the backbone model
        # num_classes=0 removes the default FC layer
        # global_pool='avg' ensures the output is a pooled feature vector (B, C)
        # drop_path_rate is used for Stochastic Depth regularization in ConvNeXt
        self.backbone = timm.create_model(
            Config.model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=Config.drop_path_rate,
        )

        # Retrieve the feature dimension size from the backbone
        in_features = self.backbone.num_features

        # Define the custom classification head
        # We use a Dropout layer followed by a Linear layer for ordinal regression
        # The output dimension is Config.num_outputs (4)
        self.head = nn.Sequential(
            nn.Dropout(p=Config.dropout_rate),
            nn.Linear(in_features, Config.num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, 4).
        """
        # Pass input through the backbone to get feature vectors
        features = self.backbone(x)

        # Pass features through the custom head to get logits
        logits = self.head(features)

        return logits
