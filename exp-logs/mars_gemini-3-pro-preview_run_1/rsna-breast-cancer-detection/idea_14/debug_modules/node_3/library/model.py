import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from library.config import Config


class DeformableAlignmentModule(nn.Module):
    """
    Module to align contralateral features to target features using dense flow prediction.
    """

    def __init__(self, in_channels):
        super(DeformableAlignmentModule, self).__init__()

        # Convolution to predict flow offsets (dx, dy) from concatenated features
        # Input: 2 * in_channels (Target + Contra)
        # Output: 2 channels (x_offset, y_offset)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=3, padding=1, bias=True),
        )

        # Initialize the final convolution weights to zero so training starts with identity transform
        nn.init.constant_(self.offset_conv[-1].weight, 0)
        nn.init.constant_(self.offset_conv[-1].bias, 0)

    def forward(self, target_feat, contra_feat):
        """
        Args:
            target_feat: (B, C, H, W)
            contra_feat: (B, C, H, W)
        Returns:
            aligned_contra_feat: (B, C, H, W)
        """
        # 1. Predict offsets
        # Concatenate along channel dimension
        concat = torch.cat([target_feat, contra_feat], dim=1)
        flow = self.offset_conv(concat)  # (B, 2, H, W)

        # 2. Create base grid
        B, C, H, W = target_feat.shape
        # Meshgrid in range [-1, 1]
        xx = torch.linspace(-1.0, 1.0, W, device=target_feat.device)
        yy = torch.linspace(-1.0, 1.0, H, device=target_feat.device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=0)  # (2, H, W)
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 2, H, W)

        # 3. Add flow to grid
        # The flow prediction needs to be scaled to the grid coordinate system [-1, 1]
        # We assume the network learns the appropriate magnitude.
        # Permute for grid_sample: (B, H, W, 2)
        sampling_grid = (base_grid + flow).permute(0, 2, 3, 1)

        # 4. Warp contralateral features
        # align_corners=True matches the generation of linspace(-1, 1)
        aligned_contra_feat = F.grid_sample(
            contra_feat,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        return aligned_contra_feat


class DeformableSiameseModel(nn.Module):
    def __init__(self, backbone_name=Config.BACKBONE, pretrained=True):
        super(DeformableSiameseModel, self).__init__()

        # 1. Backbone
        # features_only=True returns a list of feature maps
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # Get channel counts for feature levels
        # EfficientNet-B2 usually returns 5 levels. We use indices 2 (P3), 3 (P4), 4 (P5).
        # We need to dynamically check the channel dimensions.
        dummy_input = torch.randn(1, Config.IN_CHANNELS, 256, 256)
        with torch.no_grad():
            features = self.backbone(dummy_input)

        # Indices of interest: P3, P4, P5
        self.feature_indices = [2, 3, 4]
        feature_channels = [features[i].shape[1] for i in self.feature_indices]

        # 2. Alignment Modules
        self.alignment_modules = nn.ModuleList(
            [DeformableAlignmentModule(ch) for ch in feature_channels]
        )

        # 3. Classification Head
        # We concatenate GAP(Target) and GAP(Difference) for all selected levels
        # Total dims = Sum(ch * 2) over levels
        total_features = sum(ch * 2 for ch in feature_channels)

        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),  # Binary classification logits
        )

    def forward_features(self, x):
        """Extracts pyramid features from backbone."""
        all_feats = self.backbone(x)
        return [all_feats[i] for i in self.feature_indices]

    def forward(self, x_target, x_contra):
        """
        Args:
            x_target: (B, 3, H, W)
            x_contra: (B, 3, H, W)
        """
        # 1. Extract Features
        feats_target = self.forward_features(x_target)
        feats_contra = self.forward_features(x_contra)

        pooled_vectors = []

        # 2. Process each level
        for i, (f_t, f_c) in enumerate(zip(feats_target, feats_contra)):
            # Align contralateral to target
            align_mod = self.alignment_modules[i]
            f_c_aligned = align_mod(f_t, f_c)

            # Compute Difference (Symmetry Anomaly)
            diff = f_t - f_c_aligned

            # Global Average Pooling
            # Context: Target features
            # Signal: Difference features
            gap_target = F.adaptive_avg_pool2d(f_t, (1, 1)).flatten(1)
            gap_diff = F.adaptive_avg_pool2d(diff, (1, 1)).flatten(1)

            pooled_vectors.append(gap_target)
            pooled_vectors.append(gap_diff)

        # 3. Concatenate and Classify
        global_feature = torch.cat(pooled_vectors, dim=1)
        logits = self.classifier(global_feature)

        return logits
