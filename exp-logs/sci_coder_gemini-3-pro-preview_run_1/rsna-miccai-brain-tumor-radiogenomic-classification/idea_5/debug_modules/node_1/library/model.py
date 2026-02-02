import torch
import torch.nn as nn
import timm
from library.config import Config


class SiameseEfficientNet(nn.Module):
    """
    2.5D Siamese Network with a shared EfficientNet-B0 backbone.

    Architecture:
    1. Input: (Batch, Views, Channels, Height, Width)
       - Views = 3 (45%, 50%, 55% depths)
       - Channels = 3 (FLAIR, T1wCE, T2w)
    2. Backbone: Shared EfficientNet-B0 (pretrained).
    3. Aggregation: Element-wise Max-Pooling across the 'Views' dimension.
       - Logic: Preserves the strongest feature activation found in any slice.
    4. Head: Dropout -> Linear -> Logits.
    """

    def __init__(self, pretrained=True):
        super(SiameseEfficientNet, self).__init__()

        # 1. Backbone
        # Load EfficientNet-B0 from timm
        # num_classes=0 removes the classification head and returns the global pool features
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=pretrained, num_classes=0, in_chans=3
        )

        # Get the feature dimension automatically
        # For EfficientNet-B0, this is typically 1280
        self.feature_dim = self.backbone.num_features

        # 2. Classification Head
        self.dropout = nn.Dropout(p=Config.DROPOUT_RATE)
        self.fc = nn.Linear(self.feature_dim, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Views, Channels, Height, Width).
                              Views is typically 3.

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        b, v, c, h, w = x.shape

        # Flatten Batch and Views dimensions to pass through the backbone efficiently
        # New shape: (Batch * Views, Channels, Height, Width)
        x_flat = x.view(b * v, c, h, w)

        # Extract features
        # Output shape: (Batch * Views, Feature_Dim)
        features_flat = self.backbone(x_flat)

        # Reshape back to separate Batch and Views
        # Shape: (Batch, Views, Feature_Dim)
        features = features_flat.view(b, v, -1)

        # Aggregation: Element-wise Max-Pooling
        # We take the max value across the Views dimension (dim=1)
        # This acts as a "Feature-Level Maximum Intensity Projection"
        # Shape: (Batch, Feature_Dim)
        pooled_features, _ = torch.max(features, dim=1)

        # Classification Head
        # Apply dropout and linear layer
        out = self.dropout(pooled_features)
        logits = self.fc(out)

        return logits
