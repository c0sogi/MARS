import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import (
    BACKBONE,
    PRETRAINED,
    UNFREEZE_BACKBONE,
    EMBEDDING_DIM,
    TRAIN_SIGMA_FLOOR,
    SEED,
)
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(SEED)


class ParametricNet(nn.Module):
    """
    Parametric Network (ParametricNet).
    Combines a CNN backbone with a parametric tabular model.
    Reverts Spatial Attention (Cite solution_lesson_node_00030).
    """

    def __init__(self):
        super(ParametricNet, self).__init__()

        # 1. Image Backbone (EfficientNet-B0)
        # features_only=True returns a list of feature maps at different scales
        self.backbone = timm.create_model(
            BACKBONE, pretrained=PRETRAINED, features_only=True
        )

        # Determine feature channels dynamically
        # EfficientNet-B0 last feature map usually has 1280 channels
        dummy_in = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_in)
            last_feat = features[-1]
            in_channels = last_feat.shape[1]

        # 2. Pooling and Projection
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.img_projector = nn.Linear(in_channels, EMBEDDING_DIM)

        # 4. Tabular Branch
        # Inputs: Baseline(1), Age(1), Sex(1), Smoking(1) -> Total 4
        # Weeks is used in the trajectory equation, not the embedding input
        self.tab_input_dim = 4
        self.tab_projector = nn.Linear(self.tab_input_dim, EMBEDDING_DIM)
        # Note: No Batch Normalization or Layer Normalization here to preserve magnitude

        # 5. Parametric Head
        # Input: Concatenated Image + Tabular embeddings
        self.head = nn.Linear(EMBEDDING_DIM * 2, 4)
        # Outputs: alpha (baseline coeff), beta (intercept), gamma (slope), delta (uncertainty)

        # 6. Unfreezing Logic
        if UNFREEZE_BACKBONE:
            # First, freeze all backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze top blocks (blocks 5 and 6 for B0) and head layers
            # We iterate named parameters to identify them
            for name, param in self.backbone.named_parameters():
                # EfficientNet blocks are usually named 'blocks.0', 'blocks.1', etc.
                # We want the last few blocks (5 and 6) and the final conv/bn
                if (
                    "blocks.5." in name
                    or "blocks.6." in name
                    or "conv_head" in name
                    or "bn2" in name
                ):
                    param.requires_grad = True

    def forward(self, imgs, tab):
        """
        Args:
            imgs: (B, 3, H, W) - 3 slices stacked as channels
            tab: (B, 5) - [Baseline, Weeks, Age, Sex, Smoking]
        Returns:
            mu: Predicted FVC
            sigma: Predicted Confidence
        """

        # --- Image Branch ---
        # Get features from backbone (last scale)
        features = self.backbone(imgs)[-1]  # (B, C, H, W)

        # Global Average Pooling
        features = self.avgpool(features)  # (B, C, 1, 1)
        features = features.flatten(1)  # (B, C)

        # Project to embedding space
        img_emb = F.relu(self.img_projector(features))

        # --- Tabular Branch ---
        # Extract static features: Baseline(0), Age(2), Sex(3), Smoking(4)
        # Indices based on OSICDataset implementation in library/data.py
        tab_mlp_in = tab[:, [0, 2, 3, 4]]

        # Project to embedding space (Linear + ReLU)
        # No normalization layers used
        tab_emb = F.relu(self.tab_projector(tab_mlp_in))

        # --- Fusion ---
        # Late fusion via concatenation
        combined = torch.cat([img_emb, tab_emb], dim=1)

        # --- Parametric Prediction ---
        # Predict trajectory parameters
        out = self.head(combined)  # (B, 4)

        alpha = out[:, 0]
        beta = out[:, 1]
        gamma = out[:, 2]
        delta = out[:, 3]

        # --- Trajectory Logic ---
        # mu = alpha * Baseline + beta + gamma * Weeks
        baseline = tab[:, 0]
        weeks = tab[:, 1]

        mu = alpha * baseline + beta + gamma * weeks

        # --- Uncertainty Logic ---
        # sigma = softplus(delta) + floor
        sigma = F.softplus(delta) + TRAIN_SIGMA_FLOOR

        return mu, sigma
