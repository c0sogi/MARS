import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SpatialAttention(nn.Module):
    """
    Lightweight Spatial Attention Module.
    Computes a spatial mask using a 1x1 convolution and sigmoid activation,
    then scales the input feature map element-wise.
    """

    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, H, W)
        # mask shape: (B, 1, H, W)
        mask = self.sigmoid(self.conv(x))
        return x * mask


class AttentivePyramidSiamese(nn.Module):
    """
    Attentive Pyramid Symmetry-Difference Siamese Network.

    Architecture:
    1. Shared Backbone (EfficientNet-B2) extracts features at P3, P4, P5.
    2. Computes Signed Difference between Target and Contralateral features.
    3. Applies Spatial Attention to the Difference maps to suppress misalignment noise.
    4. Aggregates Global Average Pooled features from Target and Attended Difference streams.
    5. Single Linear Classification Head.
    """

    def __init__(self, backbone_name=Config.BACKBONE, pretrained=Config.PRETRAINED):
        super(AttentivePyramidSiamese, self).__init__()

        # 1. Siamese Backbone
        # features_only=True returns intermediate feature maps
        # out_indices=(2, 3, 4) corresponds to P3 (stride 8), P4 (stride 16), P5 (stride 32) for EfficientNets
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=Config.NUM_CHANNELS,  # 3: Image + Age + Implant
            features_only=True,
            out_indices=(2, 3, 4),
        )

        # Get channel counts for the extracted layers
        # feature_info.channels() returns a list of channel counts for the selected indices
        self.feature_channels = self.backbone.feature_info.channels()

        # 2. Attention Modules
        # Create a SpatialAttention module for each feature level
        self.attentions = nn.ModuleList(
            [SpatialAttention(ch) for ch in self.feature_channels]
        )

        # 3. Classification Head
        # We concatenate GAP(Target) and GAP(Difference_Attended) for each of the 3 levels.
        # Total input dim = Sum(Channel_i * 2) for i in [P3, P4, P5]
        total_embedding_dim = sum(ch * 2 for ch in self.feature_channels)

        self.classifier = nn.Linear(total_embedding_dim, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=Config.DROP_RATE)

    def forward_features(self, x):
        """Pass input through backbone to get pyramid features."""
        return self.backbone(x)

    def forward(self, x_target, x_contra):
        """
        Args:
            x_target: Tensor (B, 3, H, W) - Candidate breast image + metadata
            x_contra: Tensor (B, 3, H, W) - Contralateral breast image + metadata

        Returns:
            logits: Tensor (B, 1)
        """
        # 1. Extract Features (Siamese - Shared Weights)
        # Returns list of tensors [P3, P4, P5]
        feats_target = self.forward_features(x_target)
        feats_contra = self.forward_features(x_contra)

        pooled_vectors = []

        # 2. Multi-Scale Difference & Attention
        for i, (ft, fc) in enumerate(zip(feats_target, feats_contra)):
            # A. Compute Signed Feature Difference
            # (Age/Implant channels in input are identical, so they cancel out here,
            # removing demographic bias from the asymmetry signal)
            diff = ft - fc

            # B. Apply Spatial Attention
            # Learn to focus on tumor-like asymmetries and ignore registration errors
            diff_attended = self.attentions[i](diff)

            # C. Global Average Pooling
            # We keep the Target features (Visual Context)
            gap_target = F.adaptive_avg_pool2d(ft, 1).flatten(1)
            # We keep the Attended Difference features (Asymmetry Signal)
            gap_diff = F.adaptive_avg_pool2d(diff_attended, 1).flatten(1)

            pooled_vectors.append(gap_target)
            pooled_vectors.append(gap_diff)

        # 3. Fusion
        # Concatenate all vectors: [P3_T, P3_D, P4_T, P4_D, P5_T, P5_D]
        embedding = torch.cat(pooled_vectors, dim=1)

        embedding = self.dropout(embedding)

        # 4. Classification
        logits = self.classifier(embedding)

        return logits
