import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPRNet).

    Implements the architecture described in Lesson 52 and 60.
    Stream A (Deep): Image + Tabular -> Non-linear interaction.
    Stream B (Linear): Baseline FVC + Time -> Linear projection (Over-parameterized).
    Fusion: Summation in latent space.
    """

    def __init__(self):
        super(DSPRNet, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Image Backbone (EfficientNet-B0)
        # ---------------------------------------------------------------------
        # Cite solution_lesson_node_00054: Limit capacity to prevent overfitting
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=True,
            in_chans=3,
            num_classes=0,
            global_pool="",
        )

        num_backbone_features = self.backbone.num_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Project image features to latent dim (64)
        # Cite solution_lesson_node_00008: Project high-dim features
        self.img_project = nn.Linear(num_backbone_features, Config.EMBED_DIM)

        # Freezing Logic (Differential Learning Rate support)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze top layers
        for param in self.backbone.conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone.bn2.parameters():
            param.requires_grad = True
        for block in self.backbone.blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True

        # ---------------------------------------------------------------------
        # 2. Stream A: Deep Interaction (Non-Linear Correction)
        # ---------------------------------------------------------------------
        # Input: All tabular features (7)
        self.tabular_input_dim = 7

        # Encode tabular features
        self.deep_tabular_mlp = nn.Sequential(
            nn.Linear(self.tabular_input_dim, Config.EMBED_DIM),
            nn.ReLU(),
            nn.Linear(Config.EMBED_DIM, Config.EMBED_DIM),
            nn.ReLU(),
        )

        # Fuse Image + Tabular for Deep Stream
        # Input: 64 (Image) + 64 (Tabular) = 128
        self.deep_fusion_mlp = nn.Sequential(
            nn.Linear(Config.EMBED_DIM * 2, Config.EMBED_DIM),
            nn.ReLU(),
            nn.Linear(Config.EMBED_DIM, Config.EMBED_DIM),
        )

        # ---------------------------------------------------------------------
        # 3. Stream B: Linear Residual (Strong Baseline Prior)
        # ---------------------------------------------------------------------
        # Input: Baseline FVC, Relative Time (Indices 0, 1)
        # Cite solution_lesson_node_00052: Linear residual stream
        # Cite solution_lesson_node_00060: Over-parameterize linear stream (Project to latent)
        self.linear_input_dim = 2
        self.linear_stream_proj = nn.Linear(self.linear_input_dim, Config.EMBED_DIM)

        # ---------------------------------------------------------------------
        # 4. Head
        # ---------------------------------------------------------------------
        # Shared head for mu and sigma
        # Cite solution_lesson_node_00055: Do not isolate priors from uncertainty
        self.head = nn.Linear(Config.EMBED_DIM, 2)

    def forward(self, images, tabular):
        # --- Stream A: Deep Interaction ---
        # Image Features
        img_feat = self.backbone.forward_features(images)
        img_feat = self.global_pool(img_feat).flatten(1)
        img_embed = self.img_project(img_feat)

        # Tabular Features
        tab_deep = self.deep_tabular_mlp(tabular)

        # Concatenate and Process
        h_deep = self.deep_fusion_mlp(torch.cat([img_embed, tab_deep], dim=1))

        # --- Stream B: Linear Residual ---
        # Select Baseline FVC (0) and Time (1)
        tab_linear_in = tabular[:, 0:2]
        h_linear = self.linear_stream_proj(tab_linear_in)

        # --- Fusion ---
        # Summation in latent space
        h_final = h_deep + h_linear

        # --- Output ---
        out = self.head(h_final)

        return out
