import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPRNet).
    Cite solution_lesson_node_00052: Dual-Stream Residuals.
    Cite solution_lesson_node_00060: Over-Parameterization of Linear Baselines.

    Architecture:
    1. Stream A (Deep): EfficientNet-B0 + Clinical MLP -> Latent (64)
    2. Stream B (Prior): Linear(Base_FVC, Time) -> Latent (64)
    3. Fusion: Sum(Stream A, Stream B)
    4. Head: Linear -> mu, sigma
    """

    def __init__(self):
        super(DSPRNet, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Image Branch (Backbone)
        # ---------------------------------------------------------------------
        # Cite solution_lesson_node_00054: Use lightweight backbone (B0)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0, global_pool=""
        )
        self.num_features = self.backbone.num_features

        # Freezing Logic
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze Top Layers
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "blocks"):
            num_blocks = len(self.backbone.blocks)
            for i in range(num_blocks - 2, num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Cite solution_lesson_node_00008: Project high-dim image features
        self.img_project = nn.Sequential(
            nn.Linear(self.num_features, Config.LATENT_DIM), nn.ReLU()
        )

        # ---------------------------------------------------------------------
        # 2. Deep Stream (Image + Clinical)
        # ---------------------------------------------------------------------
        # Input: Image Latent (64) + Clinical (9)
        self.deep_fusion = nn.Sequential(
            nn.Linear(Config.LATENT_DIM + 9, 128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(128, Config.LATENT_DIM),
        )

        # ---------------------------------------------------------------------
        # 3. Prior Stream (Linear Residual)
        # ---------------------------------------------------------------------
        # Inputs: Base_FVC (idx 0), Rel_Time (idx 3)
        # Cite solution_lesson_node_00060: Over-parameterize linear stream
        self.prior_stream = nn.Linear(2, Config.LATENT_DIM)

        # ---------------------------------------------------------------------
        # 4. Prediction Head
        # ---------------------------------------------------------------------
        self.head = nn.Linear(Config.LATENT_DIM, 2)

    def forward(self, img, clin_data):
        # --- Image Features ---
        features = self.backbone.forward_features(img)
        pooled = self.global_pool(features).flatten(1)
        img_lat = self.img_project(pooled)

        # --- Deep Stream ---
        # Concatenate Image Latent + All Clinical Data
        deep_in = torch.cat([img_lat, clin_data], dim=1)
        deep_out = self.deep_fusion(deep_in)

        # --- Prior Stream ---
        # Extract Base_FVC (0) and Rel_Time (3)
        prior_in = clin_data[:, [0, 3]]
        prior_out = self.prior_stream(prior_in)

        # --- Fusion ---
        # Cite solution_lesson_node_00052: Summation fusion
        final_lat = deep_out + prior_out

        # --- Prediction ---
        out = self.head(final_lat)
        return out
