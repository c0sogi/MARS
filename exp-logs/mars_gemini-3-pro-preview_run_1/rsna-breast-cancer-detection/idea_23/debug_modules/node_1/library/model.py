import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config
from library.modules import (
    FeaturePyramidNetwork,
    DeformableAlignmentModule,
    SpatialAttentionBlock,
)


class SiameseEfficientNetFPN(nn.Module):
    """
    FPN-Enhanced Deformable Spatial-Attention Siamese Network.

    Architecture:
    1. Shared EfficientNet-B2 Backbone (P3, P4, P5).
    2. Shared Feature Pyramid Network (FPN).
    3. Per-Level Deformable Alignment of Contralateral Features.
    4. Per-Level Spatial Attention on Feature Difference.
    5. Concatenation of Context (Target) and Asymmetry (Difference) features.
    6. Classification Head.
    """

    def __init__(self, backbone_name=Config.BACKBONE, pretrained=Config.PRETRAINED):
        super().__init__()

        # 1. Backbone
        # Extract features at strides 8 (P3), 16 (P4), 32 (P5)
        # indices=(2, 3, 4) for EfficientNet corresponds to these levels
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
            in_chans=Config.CHANNELS,
            drop_rate=Config.DROP_RATE,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # Get input channel counts for the FPN from the backbone
        in_channels_list = self.backbone.feature_info.channels()

        # 2. Shared Feature Pyramid Network
        self.fpn_out_channels = 128
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list, out_channels=self.fpn_out_channels
        )

        # 3. Level-specific Modules (Alignment & Attention)
        # We have 3 levels: P3, P4, P5
        self.num_levels = 3

        self.align_modules = nn.ModuleList(
            [
                DeformableAlignmentModule(self.fpn_out_channels)
                for _ in range(self.num_levels)
            ]
        )

        self.attn_modules = nn.ModuleList(
            [
                SpatialAttentionBlock(self.fpn_out_channels)
                for _ in range(self.num_levels)
            ]
        )

        # 4. Classification Head
        # We concatenate GAP vectors for Target (Context) and Weighted Difference (Asymmetry)
        # for all 3 levels.
        # Total Features = 3 levels * (128 context + 128 asymmetry) = 768
        total_features = self.num_levels * self.fpn_out_channels * 2

        self.head = nn.Sequential(
            nn.Dropout(p=Config.DROP_RATE), nn.Linear(total_features, 1)
        )

    def forward_features(self, x):
        """
        Passes input through backbone and FPN.
        Returns list of feature maps [P3, P4, P5].
        """
        # Backbone extraction
        features = self.backbone(x)
        # FPN Fusion
        fpn_features = self.fpn(features)
        return fpn_features

    def forward(self, x_target, x_contra):
        """
        Args:
            x_target: Tensor (B, C, H, W) - Target breast image + metadata channels
            x_contra: Tensor (B, C, H, W) - Contralateral breast image + metadata channels

        Returns:
            logits: Tensor (B, 1)
        """
        # 1. Extract Features (Shared Weights)
        target_feats = self.forward_features(x_target)  # List of [P3, P4, P5]
        contra_feats = self.forward_features(x_contra)  # List of [P3, P4, P5]

        global_descriptors = []

        # 2. Process each FPN level
        for i in range(self.num_levels):
            t_feat = target_feats[i]
            c_feat = contra_feats[i]

            # A. Deformable Alignment
            # Align contralateral features to match target spatial structure
            c_aligned = self.align_modules[i](t_feat, c_feat)

            # B. Compute Difference
            # Subtraction cancels out symmetric anatomy (and demographic channels)
            diff = t_feat - c_aligned

            # C. Spatial Attention
            # Weight the difference map to suppress misalignment noise
            diff_weighted = self.attn_modules[i](diff)

            # D. Global Average Pooling
            # Capture global context from Target
            t_gap = F.adaptive_avg_pool2d(t_feat, (1, 1)).flatten(1)
            # Capture specific asymmetry signals from Difference
            d_gap = F.adaptive_avg_pool2d(diff_weighted, (1, 1)).flatten(1)

            global_descriptors.append(t_gap)
            global_descriptors.append(d_gap)

        # 3. Concatenate all features
        embedding = torch.cat(global_descriptors, dim=1)

        # 4. Classification
        logits = self.head(embedding)

        return logits
