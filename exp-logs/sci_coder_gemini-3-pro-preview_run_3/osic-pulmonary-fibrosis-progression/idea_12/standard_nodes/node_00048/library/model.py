import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import BACKBONE_NAME, FEATURE_DIM


class RSTCNet(nn.Module):
    """
    Residual-Skipped Time-Conditioned Network (RSTC-Net).

    A hybrid CNN-MLP architecture that fuses image features with tabular data using
    a Wide-and-Deep approach.

    Structure:
    1. Image Branch: EfficientNet-B2 (Top 2 blocks unfrozen) -> Projection.
    2. Deep Branch: MLP modeling interactions between Image, Tabular, and Time.
    3. Wide Branch: Identity mapping of Tabular and Time features.
    4. Head: Linear projection to mu and sigma.
    """

    def __init__(self, n_tabular_features=7):
        """
        Args:
            n_tabular_features (int): Number of tabular features (excluding time).
                                      Default is 7 based on DataProcessor (BaseFVC, Age, Sex*2, Smoke*3).
        """
        super(RSTCNet, self).__init__()

        # ==========================
        # 1. Image Branch (Backbone)
        # ==========================
        # Load backbone, remove classification head (num_classes=0 returns pooled features)
        self.backbone = timm.create_model(
            BACKBONE_NAME, pretrained=True, num_classes=0, in_chans=3
        )

        # --- Freeze / Unfreeze Logic ---
        # Freeze entire backbone first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze Head components (conv_head, bn2) if they exist
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # Unfreeze the top two convolutional blocks
        # In timm EfficientNet, blocks are stored in a ModuleList/Sequential named 'blocks'
        if hasattr(self.backbone, "blocks"):
            num_blocks = len(self.backbone.blocks)
            # Unfreeze last 2 blocks
            for i in range(num_blocks - 2, num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

        # Get backbone output dimension (num_features)
        self.n_backbone_features = self.backbone.num_features

        # Image Projection Layer
        self.img_projector = nn.Linear(self.n_backbone_features, FEATURE_DIM)

        # ==========================
        # 2. Input Dimensions
        # ==========================
        # Tabular input includes the static features + relative time
        self.input_tab_dim = n_tabular_features + 1

        # ==========================
        # 3. Deep Branch (Interaction)
        # ==========================
        # Input: Image Projection (128) + Tabular + Time
        self.deep_in_dim = FEATURE_DIM + self.input_tab_dim
        self.deep_hidden_dim = 64  # Bottleneck dimension

        # MLP: Linear -> ReLU -> Linear
        self.deep_branch = nn.Sequential(
            nn.Linear(self.deep_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.deep_hidden_dim),
        )

        # ==========================
        # 4. Wide Branch (Skip)
        # ==========================
        # Identity mapping of Tabular + Time
        self.wide_in_dim = self.input_tab_dim

        # ==========================
        # 5. Final Head
        # ==========================
        # Input: Deep Output + Wide Output
        self.final_in_dim = self.deep_hidden_dim + self.wide_in_dim

        # Predict mu and sigma
        self.head = nn.Linear(self.final_in_dim, 2)

        # Initialize head weights
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, image, tabular, time):
        """
        Args:
            image: (B, 3, H, W)
            tabular: (B, n_tabular_features)
            time: (B, 1)
        Returns:
            mu: (B, 1) Predicted FVC (standardized)
            sigma: (B, 1) Predicted Confidence
        """
        # --- Image Processing ---
        # Extract features from backbone
        # Output shape: (B, num_features)
        features = self.backbone(image)

        # Project to lower dimension
        img_embed = self.img_projector(features)  # (B, 128)

        # --- Tabular Processing ---
        # Concatenate static tabular features with time
        tab_embed = torch.cat([tabular, time], dim=1)  # (B, 8)

        # --- Deep Branch ---
        # Concatenate Image and Tabular embeddings
        deep_in = torch.cat([img_embed, tab_embed], dim=1)
        deep_out = self.deep_branch(deep_in)  # (B, 64)

        # --- Wide Branch ---
        # Pass tabular data directly
        wide_out = tab_embed  # (B, 8)

        # --- Fusion ---
        # Concatenate Deep and Wide outputs
        final_in = torch.cat([deep_out, wide_out], dim=1)

        # Predict
        raw_out = self.head(final_in)  # (B, 2)

        # Split into mu and sigma
        mu = raw_out[:, 0].unsqueeze(1)
        sigma_raw = raw_out[:, 1].unsqueeze(1)

        # Enforce positivity for sigma using Softplus
        # Add epsilon to prevent numerical instability
        sigma = F.softplus(sigma_raw) + 1e-6

        return mu, sigma
