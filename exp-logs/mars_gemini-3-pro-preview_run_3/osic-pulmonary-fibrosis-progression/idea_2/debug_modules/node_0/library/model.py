import torch
import torch.nn as nn
import timm
from library.config import Config


class OSICModel(nn.Module):
    """
    Spatially-Aware 2.5D Multi-Slice Architecture.
    Fuses multi-view image features (Apical, Middle, Basal) with clinical metadata.
    """

    def __init__(self, tabular_input_dim=8, tabular_hidden_dim=128):
        """
        Args:
            tabular_input_dim (int): Number of input tabular features. Defaults to 8.
            tabular_hidden_dim (int): Hidden dimension for tabular MLP. Defaults to 128.
        """
        super(OSICModel, self).__init__()

        # ---------------------------------------------------------------------
        # Image Branch
        # ---------------------------------------------------------------------
        # Shared Backbone: EfficientNet-B0
        # num_classes=0 returns the pooled feature vector (1280-dim for B0)
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=True, num_classes=0
        )

        # Freeze backbone weights to ensure stability and prevent overfitting
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Feature Aggregation
        # We process 3 slices independently, so the concatenated dimension is 3 * backbone_features
        self.backbone_dim = self.backbone.num_features
        self.combined_img_dim = self.backbone_dim * 3

        # Projection Layer
        # Compresses high-dim features (3840) to compact embedding (128)
        self.img_projector = nn.Linear(self.combined_img_dim, Config.IMG_EMBED_DIM)

        # ---------------------------------------------------------------------
        # Tabular Branch
        # ---------------------------------------------------------------------
        # MLP without Batch Normalization (as per lessons learned)
        # Structure: Linear -> ReLU -> Linear -> ReLU
        self.tab_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, tabular_hidden_dim),
            nn.ReLU(),
            nn.Linear(tabular_hidden_dim, tabular_hidden_dim),
            nn.ReLU(),
        )

        # ---------------------------------------------------------------------
        # Fusion & Head
        # ---------------------------------------------------------------------
        # Concatenate Image Embedding + Tabular Embedding
        self.fusion_dim = Config.IMG_EMBED_DIM + tabular_hidden_dim

        # Final Regression Head
        # Outputs: FVC_pred (mu) and Confidence (sigma)
        # Dropout is removed to preserve strong linear signals
        self.head = nn.Linear(self.fusion_dim, 2)

    def forward(self, image, tabular):
        """
        Args:
            image (torch.Tensor): Shape (B, 3, H, W). The 3 channels represent
                                  Apical, Middle, and Basal slices.
            tabular (torch.Tensor): Shape (B, 8). Clinical features.

        Returns:
            torch.Tensor: Shape (B, 2). [FVC_pred, Confidence]
        """
        # --- Image Processing ---
        # Process each slice independently through the shared backbone
        slice_features = []

        # Iterate over the 3 slices (channels)
        for i in range(3):
            # Extract single slice: (B, 1, H, W)
            # We slice [:, i:i+1, ...] to keep the channel dim
            slice_img = image[:, i : i + 1, :, :]

            # Repeat to 3 channels for ImageNet-pretrained backbone: (B, 3, H, W)
            slice_rgb = slice_img.repeat(1, 3, 1, 1)

            # Extract features: (B, 1280)
            feat = self.backbone(slice_rgb)
            slice_features.append(feat)

        # Concatenate features from all 3 slices: (B, 3840)
        # This preserves the vertical spatial information (Top vs Bottom)
        img_concat = torch.cat(slice_features, dim=1)

        # Project to lower dimension: (B, 128)
        img_embed = self.img_projector(img_concat)

        # --- Tabular Processing ---
        # Process clinical data: (B, 128)
        tab_embed = self.tab_mlp(tabular)

        # --- Fusion ---
        # Concatenate embeddings: (B, 256)
        fused = torch.cat([img_embed, tab_embed], dim=1)

        # --- Prediction ---
        # Predict Mu and Sigma: (B, 2)
        out = self.head(fused)

        return out
