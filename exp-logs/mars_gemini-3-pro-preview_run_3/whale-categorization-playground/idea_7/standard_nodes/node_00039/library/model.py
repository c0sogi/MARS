import torch
import torch.nn as nn
import timm
from library.config import Config


class WhaleModel(nn.Module):
    """
    Whale Identification Model.

    Implements a backbone-based feature extractor with a specific projection head
    designed for ArcFace-based metric learning.

    Architecture:
        - Backbone: Configurable via timm (e.g., EfficientNet-B2, EfficientNet-B3).
        - Pooling: Global Average Pooling (GAP).
        - Head: Batch Normalization -> Dropout -> Linear -> Batch Normalization.
    """

    def __init__(self, model_name, pretrained=True):
        """
        Initialize the WhaleModel.

        Args:
            model_name (str): The name of the timm backbone to use (e.g., 'efficientnet_b2').
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(WhaleModel, self).__init__()

        # Create the backbone using timm
        # num_classes=0 removes the default classification head
        # global_pool='' ensures we get the spatial feature maps (B, C, H, W)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input features for the head
        # Most timm models have a num_features attribute
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            # Fallback: Run a dummy forward pass to infer shape
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
                features = self.backbone(dummy_input)
                in_features = features.shape[1]

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Projection Head
        # As per strategy: BN -> Dropout -> Linear -> BN
        # This structure helps stabilize training with ArcFace
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.2),
            nn.Linear(in_features, Config.EMBEDDING_DIM),
            nn.BatchNorm1d(Config.EMBEDDING_DIM),
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Feature embeddings of shape (Batch, Embedding_Dim).
        """
        # 1. Feature Extraction
        features = self.backbone(x)  # Output: (B, C, H, W)

        # 2. Pooling
        x = self.global_pool(features)  # Output: (B, C, 1, 1)
        x = x.flatten(1)  # Output: (B, C)

        # 3. Projection Head
        embeddings = self.head(x)  # Output: (B, Embedding_Dim)

        return embeddings
