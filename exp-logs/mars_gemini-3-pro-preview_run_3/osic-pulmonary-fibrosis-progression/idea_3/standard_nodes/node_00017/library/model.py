import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class MultiViewNet(nn.Module):
    """
    Content-Adaptive 2.5D Multi-View Network (Idea 3).

    Architecture:
    - Image Branch: Frozen EfficientNet-B0 extracting features from 3 slices (Apical, Middle, Basal).
      Features are concatenated and projected to a lower dimension.
    - Tabular Branch: Simple MLP (Linear -> ReLU) without Batch Norm.
    - Head: Concatenation of branches -> Linear Regression Head -> (Mu, Sigma).
    """

    def __init__(self):
        super(MultiViewNet, self).__init__()

        # ---------------------------------------------------------------------
        # Image Branch
        # ---------------------------------------------------------------------
        # Load pre-trained EfficientNet-B0
        # num_classes=0 returns the pooled feature vector (1280 dim) instead of logits
        self.backbone = timm.create_model(
            Config.backbone_name, pretrained=Config.pretrained, num_classes=0
        )

        # Freeze backbone weights as per strategy
        if Config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection Head
        # Input: 3 slices * 1280 features = 3840
        # Output: Projected dimension (e.g., 128)
        # Added LayerNorm (Cite Lesson 10) to stabilize fusion
        self.img_projection = nn.Sequential(
            nn.Linear(Config.combined_feature_dim, Config.projection_dim),
            nn.LayerNorm(Config.projection_dim),
            nn.ReLU(),
        )

        # ---------------------------------------------------------------------
        # Tabular Branch
        # ---------------------------------------------------------------------
        # MLP with LayerNorm (Cite Lesson 8, 10)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(Config.tabular_input_dim, Config.tabular_hidden_dim),
            nn.LayerNorm(Config.tabular_hidden_dim),
            nn.ReLU(),
        )

        # ---------------------------------------------------------------------
        # Prediction Head
        # ---------------------------------------------------------------------
        # Input: Projected Image Features + Tabular Features
        head_input_dim = Config.projection_dim + Config.tabular_hidden_dim

        # Output: 2 values (Standardized Mean FVC, Standardized Confidence Sigma)
        # No Dropout in final layer to preserve linear relationships
        self.head = nn.Linear(head_input_dim, 2)

    def forward(self, images, tabular):
        """
        Forward pass of the network.

        Args:
            images (torch.Tensor): Shape (B, 3, H, W). The 3 selected lung slices.
            tabular (torch.Tensor): Shape (B, 5). The clinical features.

        Returns:
            mu (torch.Tensor): Predicted standardized FVC mean (B,).
            sigma (torch.Tensor): Predicted standardized confidence (B,).
        """
        batch_size = images.size(0)

        # --- Image Branch ---
        # 1. Reshape to process each slice individually through the backbone
        # Input: (B, 3, H, W) -> (B*3, 1, H, W)
        x = images.view(
            batch_size * Config.num_slices, 1, Config.img_size, Config.img_size
        )

        # 2. Expand to 3 channels for EfficientNet (expects RGB)
        # (B*3, 1, H, W) -> (B*3, 3, H, W)
        x = x.repeat(1, 3, 1, 1)

        # 3. Extract features using frozen backbone
        # Output: (B*3, 1280)
        features = self.backbone(x)

        # 4. Reshape back to batch format and concatenate slice features
        # (B*3, 1280) -> (B, 3 * 1280) = (B, 3840)
        features = features.view(batch_size, -1)

        # 5. Project to lower dimension to prevent overwhelming tabular data
        # (B, 3840) -> (B, 128)
        img_out = self.img_projection(features)

        # --- Tabular Branch ---
        # Process clinical features
        # (B, 5) -> (B, 64)
        tab_out = self.tabular_mlp(tabular)

        # --- Fusion & Prediction ---
        # Concatenate image and tabular features
        # (B, 128 + 64) -> (B, 192)
        combined = torch.cat([img_out, tab_out], dim=1)

        # Final regression
        # (B, 192) -> (B, 2)
        output = self.head(combined)

        # Split output into mean (mu) and raw confidence
        mu = output[:, 0]
        raw_sigma = output[:, 1]

        # Enforce positivity and numerical stability for sigma
        # Using softplus + epsilon to avoid zero or negative standard deviation
        sigma = F.softplus(raw_sigma) + 1e-3

        return mu, sigma
