import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiPlanarSiameseNet(nn.Module):
    """
    Multi-Planar 2.5D Holographic Network (Siamese Architecture).

    This network processes three orthogonal views (Axial, Coronal, Sagittal) of the brain
    using a shared EfficientNet backbone. The resulting feature vectors are concatenated
    to preserve spatial context from all three planes before being passed to a final
    classification head.
    """

    def __init__(self, backbone_name="efficientnet_b0", pretrained=True):
        """
        Args:
            backbone_name (str): Name of the timm model to use as backbone.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(MultiPlanarSiameseNet, self).__init__()

        # Shared Backbone
        # We use num_classes=0 to remove the top classification layer and get the global pooled features.
        # in_chans is set to Config.NUM_CHANNELS (3) to match the composite input (FLAIR, T1wCE, T2w).
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=Config.NUM_CHANNELS,
            global_pool="avg",
        )

        # Retrieve the feature dimension size from the backbone (e.g., 1280 for EfficientNet-B0)
        self.feature_dim = self.backbone.num_features

        # Fusion Head
        # We concatenate the features from the 3 views, so the input dimension is 3 * feature_dim.
        # Output is 1 logit for binary classification.
        self.classifier = nn.Linear(self.feature_dim * 3, 1)

    def forward_one_view(self, x):
        """
        Passes a single view through the shared backbone.
        """
        return self.backbone(x)

    def forward(self, axial, coronal, sagittal):
        """
        Forward pass for the Siamese Network.

        Args:
            axial (torch.Tensor): Batch of Axial views (B, C, H, W)
            coronal (torch.Tensor): Batch of Coronal views (B, C, H, W)
            sagittal (torch.Tensor): Batch of Sagittal views (B, C, H, W)

        Returns:
            torch.Tensor: Logits (B, 1)
        """
        # Extract features for each view using the shared backbone
        # Shape: (Batch_Size, Feature_Dim)
        feat_ax = self.forward_one_view(axial)
        feat_cor = self.forward_one_view(coronal)
        feat_sag = self.forward_one_view(sagittal)

        # Concatenate features
        # Shape: (Batch_Size, Feature_Dim * 3)
        combined_features = torch.cat([feat_ax, feat_cor, feat_sag], dim=1)

        # Final classification
        # Shape: (Batch_Size, 1)
        logits = self.classifier(combined_features)

        return logits
