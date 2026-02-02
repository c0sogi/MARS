import torch
import torch.nn as nn
import timm
from library.config import Config


class TriSlabModel(nn.Module):
    """
    Tri-Slab Depth-Encoded Network with Dynamic Uncertainty.

    Components:
    1. Visual Backbone: EfficientNet-B0 processing 3-channel Tri-Slab MIPs.
    2. Tabular Branch: MLP processing clinical metadata.
    3. Parametric Head: Predicts trajectory slope and uncertainty parameters.
    """

    def __init__(self, cfg=Config):
        super(TriSlabModel, self).__init__()

        # 1. Visual Backbone
        # Using timm to create EfficientNet-B0
        # num_classes=0 removes the classifier, returning the pooled features
        self.backbone = timm.create_model(
            cfg.BACKBONE,
            pretrained=cfg.PRETRAINED,
            num_classes=0,
            in_chans=cfg.IN_CHANNELS,
        )

        # Get the number of features output by the backbone (1280 for EfficientNet-B0)
        self.vis_feature_dim = self.backbone.num_features

        # 2. Tabular Branch
        # Input: 4 features (Age, Percent, Sex, SmokingStatus)
        # We project this to a reasonable embedding size (e.g., 128)
        self.tab_input_dim = 4
        self.tab_hidden_dim = 128

        self.tabular_mlp = nn.Sequential(
            nn.Linear(self.tab_input_dim, self.tab_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.tab_hidden_dim, self.tab_hidden_dim),
            nn.ReLU(),
        )

        # 3. Fusion Head
        # Concatenate Visual (1280) + Tabular (128) -> Output (3)
        self.fusion_dim = self.vis_feature_dim + self.tab_hidden_dim

        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, cfg.NUM_OUTPUTS),
            # Output: [alpha, sigma_base, sigma_growth]
        )

    def forward(self, images, tabular):
        """
        Args:
            images (Tensor): (B, 3, H, W) Tri-Slab MIP images.
            tabular (Tensor): (B, 4) Normalized clinical features.

        Returns:
            Tensor: (B, 3) Predicted parameters.
        """
        # Visual Features
        vis_feats = self.backbone(images)  # (B, 1280)

        # Tabular Features
        tab_feats = self.tabular_mlp(tabular)  # (B, 128)

        # Fusion
        concat_feats = torch.cat([vis_feats, tab_feats], dim=1)  # (B, 1408)

        # Prediction
        output = self.head(concat_feats)  # (B, 3)

        return output
