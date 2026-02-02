import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DeepTabularAligner(nn.Module):
    """
    Projects low-dimensional clinical metadata into a high-dimensional semantic manifold.
    Structure: Linear -> GELU -> Linear -> GELU -> Linear
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DAVRNet(nn.Module):
    """
    Deep-Aligned Visual-Residual Dual-Axis Network (DAVR-Net).

    Architecture:
    1. Dual Independent EfficientNet-B0 Backbones (Axial & Coronal).
    2. Deep Tabular Alignment MLP.
    3. Pre-Norm Symmetric Self-Attention for Fusion.
    4. Visual-Exclusive Pooled Readout.
    5. Prior-Anchored Parametric Head (Alpha, Sigma_Base, Sigma_Growth).
    """

    def __init__(self):
        super().__init__()

        # ==========================================
        # 1. Independent High-Fidelity Visual Backbones
        # ==========================================
        # Branch A: Axial View
        self.backbone_ax = timm.create_model(
            Config.backbone_name,
            pretrained=Config.backbone_pretrained,
            num_classes=0,  # Returns pooled global features
            global_pool="avg",
        )

        # Branch B: Coronal View
        self.backbone_cor = timm.create_model(
            Config.backbone_name,
            pretrained=Config.backbone_pretrained,
            num_classes=0,
            global_pool="avg",
        )

        self.visual_dim = Config.backbone_out_dim  # 1280 for EfficientNet-B0

        # ==========================================
        # 2. Deep Tabular Alignment
        # ==========================================
        self.tabular_aligner = DeepTabularAligner(
            input_dim=Config.tabular_input_dim, output_dim=Config.tabular_hidden_dim
        )

        # ==========================================
        # 3. Pre-Norm Symmetric Attention
        # ==========================================
        # Using TransformerEncoderLayer to fuse [Axial, Coronal, Tabular]
        # norm_first=True enables Pre-Norm architecture for stability
        self.attention = nn.TransformerEncoderLayer(
            d_model=self.visual_dim,
            nhead=Config.num_attention_heads,
            dim_feedforward=self.visual_dim * 2,
            dropout=Config.dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # ==========================================
        # 4. Parametric Head
        # ==========================================
        # Input: Contextualized Visual Residual (1280) + Raw Tabular (7)
        head_input_dim = self.visual_dim + Config.tabular_input_dim

        self.head_alpha = nn.Linear(head_input_dim, 1)
        self.head_sigma_base = nn.Linear(head_input_dim, 1)
        self.head_sigma_growth = nn.Linear(head_input_dim, 1)

        # Normalization constants for on-the-fly normalization
        self.register_buffer("mean", torch.tensor(Config.mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(Config.std).view(1, 3, 1, 1))

    def normalize_images(self, x):
        """Normalize images to ImageNet stats."""
        return (x - self.mean) / self.std

    def forward(self, axial_img, coronal_img, tabular):
        """
        Args:
            axial_img: (B, 3, 224, 224) - [0, 1]
            coronal_img: (B, 3, 224, 224) - [0, 1]
            tabular: (B, 7) - Normalized clinical features

        Returns:
            alpha: (B, 1) - Slope
            sigma_base: (B, 1) - Base confidence
            sigma_growth: (B, 1) - Growth confidence
        """
        # 0. Normalize Images
        axial_img = self.normalize_images(axial_img)
        coronal_img = self.normalize_images(coronal_img)

        # 1. Extract Visual Features (High-Fidelity)
        # Shape: (B, 1280)
        v_ax = self.backbone_ax(axial_img)
        v_cor = self.backbone_cor(coronal_img)

        # 2. Align Tabular Features
        # Shape: (B, 1280)
        v_tab = self.tabular_aligner(tabular)

        # 3. Stack for Attention
        # Sequence: [Axial, Coronal, Tabular] -> Shape: (B, 3, 1280)
        seq = torch.stack([v_ax, v_cor, v_tab], dim=1)

        # 4. Contextualization (Symmetric Attention)
        # Shape: (B, 3, 1280)
        seq_out = self.attention(seq)

        # 5. Visual-Exclusive Readout
        # Extract updated visual tokens only (discard tabular token)
        v_ax_prime = seq_out[:, 0, :]
        v_cor_prime = seq_out[:, 1, :]

        # Average Pooling to get Visual Residual
        # Shape: (B, 1280)
        v_res = (v_ax_prime + v_cor_prime) / 2.0

        # 6. Prior-Anchored Parametric Head
        # Skip connection: Concatenate Visual Residual with Raw Tabular
        # Shape: (B, 1280 + 7)
        head_in = torch.cat([v_res, tabular], dim=1)

        # Predict Parameters
        alpha = self.head_alpha(head_in)

        # Apply Softplus to ensure positive confidence values
        sigma_base = F.softplus(self.head_sigma_base(head_in))
        sigma_growth = F.softplus(self.head_sigma_growth(head_in))

        return alpha, sigma_base, sigma_growth
