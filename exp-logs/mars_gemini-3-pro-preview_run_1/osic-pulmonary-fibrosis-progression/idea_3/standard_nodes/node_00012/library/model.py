import torch
import torch.nn as nn
import timm


class DualAxisTriSlabModel(nn.Module):
    """
    Dual-Axis Tri-Slab Network for Lung Function Decline Prediction.

    This model processes two orthogonal views (Axial and Coronal) of a CT scan
    using parallel CNN backbones. The visual features are fused with clinical
    tabular features to predict the trajectory of FVC decline.

    Architecture:
    - Axial Branch: EfficientNet-B0 (Channel-encoded depth)
    - Coronal Branch: EfficientNet-B0 (Spatial-encoded depth)
    - Tabular Branch: MLP
    - Head: Linear Regression -> [alpha, sigma_base, sigma_growth]
    """

    def __init__(
        self,
        backbone_name="efficientnet_b0",
        pretrained=True,
        tabular_input_dim=5,
        tabular_hidden_dim=128,
        output_dim=3,
    ):
        """
        Args:
            backbone_name (str): Name of the timm backbone to use.
            pretrained (bool): Whether to load pretrained ImageNet weights.
            tabular_input_dim (int): Number of input tabular features (Age, Sex, Smoking).
            tabular_hidden_dim (int): Hidden dimension for the tabular MLP.
            output_dim (int): Number of output scalars (alpha, sigma_base, sigma_growth).
        """
        super(DualAxisTriSlabModel, self).__init__()

        # 1. Axial Branch (Top-down view, Z-axis encoded in channels)
        self.axial_backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Return pooled features
            global_pool="avg",  # Ensure global average pooling
        )

        # 2. Coronal Branch (Front-back view, Z-axis is spatial vertical)
        self.coronal_backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Retrieve feature dimension (e.g., 1280 for EfficientNet-B0)
        self.backbone_dim = self.axial_backbone.num_features

        # 3. Tabular Branch (Clinical Data)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, tabular_hidden_dim),
            nn.ReLU(),
            nn.Linear(tabular_hidden_dim, tabular_hidden_dim),
            nn.ReLU(),
        )

        # 4. Fusion and Prediction Head
        # Concatenate: Axial features + Coronal features + Tabular features
        fusion_dim = (self.backbone_dim * 2) + tabular_hidden_dim

        self.head = nn.Linear(fusion_dim, output_dim)

    def forward(self, axial_img, coronal_img, tabular):
        """
        Args:
            axial_img (torch.Tensor): (B, 3, H, W) - Axial Tri-Slab images.
            coronal_img (torch.Tensor): (B, 3, H, W) - Coronal Tri-Slab images.
            tabular (torch.Tensor): (B, 5) - Normalized clinical features.

        Returns:
            torch.Tensor: (B, 3) - Predicted trajectory parameters:
                          [alpha (slope), sigma_base, sigma_growth]
        """
        # Extract Visual Features
        axial_feats = self.axial_backbone(axial_img)  # (B, backbone_dim)
        coronal_feats = self.coronal_backbone(coronal_img)  # (B, backbone_dim)

        # Extract Tabular Features
        tabular_feats = self.tabular_mlp(tabular)  # (B, tabular_hidden_dim)

        # Feature Fusion
        fused_feats = torch.cat([axial_feats, coronal_feats, tabular_feats], dim=1)

        # Prediction
        # Output is [alpha, sigma_base, sigma_growth]
        # We output raw linear values. Constraints (e.g. positivity for sigma)
        # are handled by the loss function or metric logic.
        out = self.head(fused_feats)

        return out
