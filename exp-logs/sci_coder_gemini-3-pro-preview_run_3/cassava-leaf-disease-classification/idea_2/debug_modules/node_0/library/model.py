import torch
import torch.nn as nn
import timm
from library.config import CFG


class CassavaModel(nn.Module):
    """
    Cassava Leaf Disease Classification Model.

    Architecture:
    - Backbone: EfficientNet-B4 (via timm) initialized with NoisyStudent weights.
    - Head: Global Average Pooling -> Dropout -> Linear Layer.
    """

    def __init__(self, model_name=CFG.model_name, pretrained=True):
        """
        Initialize the model.

        Args:
            model_name (str): The name of the timm model to load. Defaults to CFG.model_name.
            pretrained (bool): Whether to load pre-trained weights. Defaults to True.
        """
        super(CassavaModel, self).__init__()

        # Load the backbone using timm
        # num_classes=0 removes the original classification head
        # global_pool='avg' ensures the output is the result of Global Average Pooling
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Retrieve the number of input features for the final layer
        in_features = self.backbone.num_features

        # Define the custom classification head
        # As per requirements: GAP (in backbone) -> Dropout -> Linear
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features, CFG.num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of images (B, C, H, W).

        Returns:
            torch.Tensor: Raw logits (B, num_classes).
        """
        # Pass through backbone (includes Global Average Pooling)
        features = self.backbone(x)

        # Apply Dropout
        x = self.dropout(features)

        # Final Linear Layer
        logits = self.fc(x)

        return logits
