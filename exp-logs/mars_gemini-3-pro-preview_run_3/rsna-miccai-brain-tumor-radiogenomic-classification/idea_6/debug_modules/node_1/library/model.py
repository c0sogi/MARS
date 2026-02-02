import torch
import torch.nn as nn
import timm
from library.config import Config


class DualStreamSiameseNet(nn.Module):
    """
    A Siamese 2.5D Convolutional Neural Network for MGMT promoter methylation prediction.

    Architecture:
    - Backbone: EfficientNet-B0 (Shared weights)
    - Input: Two streams (Even slices, Odd slices), each with 64 channels.
    - Fusion: Late fusion via concatenation of Global Average Pooled features.
    - Head: Single linear layer for binary classification.
    """

    def __init__(self):
        super(DualStreamSiameseNet, self).__init__()

        # Initialize the shared backbone
        # We use EfficientNet-B0 pre-trained on ImageNet.
        # in_chans=64 adapts the first layer to accept our 16 slices * 4 modalities input.
        # num_classes=0 removes the classification head, returning the pooled feature vector.
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            in_chans=Config.INPUT_CHANNELS,
            num_classes=0,
            global_pool="avg",
        )

        # Get the number of output features from the backbone (e.g., 1280 for EfficientNet-B0)
        num_features = self.backbone.num_features

        # Fusion Head
        # We concatenate the feature vectors from both streams, so input dim is * 2.
        # Output is 1 logit (for BCEWithLogitsLoss).
        self.fc = nn.Linear(num_features * 2, 1)

    def forward(self, x_even, x_odd):
        """
        Forward pass for the Siamese Network.

        Args:
            x_even (torch.Tensor): Input tensor for Even stream. Shape (B, 64, H, W).
            x_odd (torch.Tensor): Input tensor for Odd stream. Shape (B, 64, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # Pass both streams through the shared backbone
        # Output shape: (Batch_Size, num_features)
        feat_even = self.backbone(x_even)
        feat_odd = self.backbone(x_odd)

        # Late Fusion: Concatenate features
        # Output shape: (Batch_Size, num_features * 2)
        combined = torch.cat([feat_even, feat_odd], dim=1)

        # Final Classification
        # Output shape: (Batch_Size, 1)
        logits = self.fc(combined)

        return logits
