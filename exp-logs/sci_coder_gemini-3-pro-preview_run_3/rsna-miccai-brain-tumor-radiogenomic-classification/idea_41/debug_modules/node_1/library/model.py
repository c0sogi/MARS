import torch
import torch.nn as nn
import timm
from library.config import BACKBONE, IN_CHANS, NUM_CLASSES, DROP_PATH_RATE


class SiameseEfficientNet(nn.Module):
    """
    Siamese Dual-View Native-Resolution (SDV-NR) Network.

    This architecture processes two spatially interleaved views (Even and Odd slices)
    of a 3D MRI volume using a shared 2.5D CNN backbone.

    Architecture:
    1. Shared Backbone: EfficientNet-B0 (pretrained on ImageNet).
       - Input channels: 64 (16 slices * 4 modalities).
       - Output: Global Average Pooled feature vector.
    2. Fusion: Late fusion via concatenation of the two feature vectors.
    3. Classifier: Single fully connected layer.
    """

    def __init__(self):
        super(SiameseEfficientNet, self).__init__()

        # Initialize the shared backbone using timm
        # num_classes=0 removes the classification head and returns the pooled features
        self.backbone = timm.create_model(
            BACKBONE,
            pretrained=True,
            in_chans=IN_CHANS,
            num_classes=0,
            drop_path_rate=DROP_PATH_RATE,
        )

        # Retrieve the feature dimension size from the backbone
        # For EfficientNet-B0, this is typically 1280
        self.num_features = self.backbone.num_features

        # Define the classifier head
        # We concatenate the features from both views, so input dim is num_features * 2
        self.classifier = nn.Linear(self.num_features * 2, NUM_CLASSES)

    def forward(self, x_even, x_odd):
        """
        Forward pass of the Siamese network.

        Args:
            x_even (torch.Tensor): Tensor of shape (B, 64, H, W) representing the Even view.
            x_odd (torch.Tensor): Tensor of shape (B, 64, H, W) representing the Odd view.

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # Pass both views through the shared backbone
        # Output shape for each: (B, num_features)
        feat_even = self.backbone(x_even)
        feat_odd = self.backbone(x_odd)

        # Late Fusion: Concatenate the feature vectors along the channel dimension
        # Shape: (B, num_features * 2)
        combined_features = torch.cat([feat_even, feat_odd], dim=1)

        # Classification
        # Shape: (B, 1)
        logits = self.classifier(combined_features)

        return logits
