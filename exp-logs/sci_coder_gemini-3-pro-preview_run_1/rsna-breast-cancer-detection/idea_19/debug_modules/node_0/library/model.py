import torch
import torch.nn as nn
import timm

from library.config import BACKBONE, DROP_RATE, NUM_CLASSES, PRETRAINED


class PyramidSiameseEfficientNet(nn.Module):
    """
    Pyramid Symmetry-Difference Siamese Network.

    Architecture:
    1. Backbone: EfficientNet-B2 (Shared Weights).
    2. Inputs: Target Image (3ch) and Contralateral Image (3ch).
    3. Feature Extraction: Extracts features at stages P3 (stride 8), P4 (stride 16), and P5 (stride 32).
    4. Difference Module: Computes Signed Feature Difference (Target - Contra) at each stage.
    5. Fusion: Global Average Pooling on both Target features and Difference features.
    6. Head: Concatenates all pooled vectors and passes through a linear classifier.
    """

    def __init__(self):
        super(PyramidSiameseEfficientNet, self).__init__()

        # Initialize backbone with multi-scale feature extraction
        # out_indices=(2, 3, 4) corresponds to P3, P4, P5 for EfficientNet
        self.backbone = timm.create_model(
            BACKBONE,
            pretrained=PRETRAINED,
            features_only=True,
            out_indices=(2, 3, 4),
            in_chans=3,
            drop_rate=DROP_RATE,
        )

        # Get channel counts for the extracted feature maps
        # feature_info.channels() returns a list of channel counts for the selected indices
        self.feature_channels = self.backbone.feature_info.channels()

        # Calculate total dimension for the linear head
        # We concat GAP(Target) and GAP(Difference) for each of the 3 stages
        # Total Dim = 2 * (C_P3 + C_P4 + C_P5)
        total_channels = sum(self.feature_channels) * 2

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        self.head = nn.Sequential(
            nn.Dropout(p=DROP_RATE), nn.Linear(total_channels, NUM_CLASSES)
        )

    def forward_features(self, x):
        """
        Extracts multi-scale features from the backbone.
        Returns a list of tensors [P3, P4, P5].
        """
        return self.backbone(x)

    def forward(self, target, contra):
        """
        Args:
            target (torch.Tensor): Target breast image batch (B, 3, H, W).
            contra (torch.Tensor): Contralateral breast image batch (B, 3, H, W).

        Returns:
            torch.Tensor: Logits (B, 1).
        """
        # 1. Extract Features (Shared Weights)
        # Each output is a list of tensors: [feat_p3, feat_p4, feat_p5]
        target_feats = self.forward_features(target)
        contra_feats = self.forward_features(contra)

        embeddings = []

        # 2. Process each scale
        for t_feat, c_feat in zip(target_feats, contra_feats):
            # a. Compute Signed Feature Difference
            # D_i = F_{target, i} - F_{contra, i}
            diff_feat = t_feat - c_feat

            # b. Global Average Pooling
            # Flatten to (B, C)
            t_pool = self.global_pool(t_feat).flatten(1)
            d_pool = self.global_pool(diff_feat).flatten(1)

            # Collect vectors
            embeddings.append(t_pool)
            embeddings.append(d_pool)

        # 3. Concatenate all vectors
        # Shape: (B, Total_Channels)
        final_embedding = torch.cat(embeddings, dim=1)

        # 4. Classification
        logits = self.head(final_embedding)

        return logits
