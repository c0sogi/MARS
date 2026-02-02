import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class VisualBackbone(nn.Module):
    """
    Independent Low-Capacity Visual Backbone.
    Uses EfficientNet-B0 initialized with ImageNet weights.
    Outputs the native 1280-dimensional feature vector (Global Average Pooled).
    """

    def __init__(self):
        super().__init__()
        # Load EfficientNet-B0, remove classifier (num_classes=0)
        # global_pool='avg' ensures we get the (B, 1280) vector
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

    def forward(self, x):
        # Input: (B, 3, 224, 224)
        # Output: (B, 1280)
        return self.backbone(x)


class TabularEncoder(nn.Module):
    """
    Shared-Latent Tabular Encoder.
    Projects raw metadata to a robust 128-dimensional latent vector.
    """

    def __init__(self):
        super().__init__()
        # Input: 4 features [Percent, Age, Sex, Smoking]
        # Output: 128 latent dim (Config.LATENT_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, Config.LATENT_DIM),
            nn.GELU(),
        )

    def forward(self, x):
        # Input: (B, 4)
        # Output: (B, 128)
        return self.mlp(x)


class BBSLNet(nn.Module):
    """
    Balanced-Bottleneck Shared-Latent Network (BBSL-Net).

    Key Features:
    1. Independent processing of Axial and Coronal views.
    2. Shared Latent representation for clinical metadata.
    3. Symmetric Attention Fusion with Pre-Norm.
    4. Balanced Bottleneck Head strictly enforcing 50/50 signal parity
       between Visual Context and Clinical Prior.
    """

    def __init__(self):
        super().__init__()

        # 1. Independent Visual Backbones
        self.backbone_ax = VisualBackbone()
        self.backbone_cor = VisualBackbone()

        # 2. Shared-Latent Tabular Encoder
        self.tab_encoder = TabularEncoder()

        # 3. Fusion Components
        # Project latent (128) to visual dim (1280) for attention alignment
        self.tab_projection = nn.Linear(Config.LATENT_DIM, Config.VISUAL_DIM)
        # LayerNorm to prevent initialization shock from the projection
        self.tab_norm = nn.LayerNorm(Config.VISUAL_DIM)

        # Transformer Encoder Layer for Fusion
        # Pre-Norm, Batch First, Single Layer
        self.fusion_layer = nn.TransformerEncoderLayer(
            d_model=Config.VISUAL_DIM,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-Norm configuration
        )

        # 4. Balanced-Bottleneck Head
        # Compress fused visual context (1280) -> 128
        self.bottleneck = nn.Linear(Config.VISUAL_DIM, Config.LATENT_DIM)

        # Final Prediction Head
        # Input: 128 (Compressed Visual) + 128 (Shared Latent) = 256
        # Output: 3 parameters (alpha, sigma_base, sigma_growth)
        self.head = nn.Linear(Config.LATENT_DIM * 2, 3)

    def forward(self, img_ax, img_cor, meta):
        """
        Args:
            img_ax: (B, 3, 224, 224) Axial Tri-Slab
            img_cor: (B, 3, 224, 224) Coronal Tri-Slab
            meta: (B, 4) Normalized metadata
        Returns:
            (B, 3) tensor containing [alpha, sigma_base, sigma_growth]
        """
        # --- Visual Extraction ---
        # (B, 1280)
        v_ax = self.backbone_ax(img_ax)
        v_cor = self.backbone_cor(img_cor)

        # --- Tabular Extraction ---
        # (B, 128) -> T_lat
        t_lat = self.tab_encoder(meta)

        # --- Fusion Preparation ---
        # Flow A: Project tabular to visual dim for fusion
        # (B, 1280)
        t_align = self.tab_projection(t_lat)
        t_align = self.tab_norm(t_align)

        # Stack tokens: [Axial, Coronal, Tabular] -> (B, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)

        # --- Attention Fusion ---
        # (B, 3, 1280)
        fused_tokens = self.fusion_layer(tokens)

        # Holistic Readout: Global Average Pooling across all tokens
        # (B, 1280) -> H_fused
        h_fused = fused_tokens.mean(dim=1)

        # --- Balanced Bottleneck ---
        # Compress visual context: (B, 1280) -> (B, 128)
        h_compressed = self.bottleneck(h_fused)

        # Concatenate with original Shared Latent: (B, 128 + 128) -> (B, 256)
        # Flow B: T_lat is preserved and concatenated here
        h_final = torch.cat([h_compressed, t_lat], dim=1)

        # --- Prediction ---
        # (B, 3) -> [alpha, sigma_base, sigma_growth]
        out = self.head(h_final)

        # Apply Softplus to sigmas (indices 1 and 2) to ensure positivity
        # Alpha (slope) at index 0 remains linear (can be negative)
        alpha = out[:, 0].unsqueeze(1)
        sigmas = F.softplus(out[:, 1:])

        return torch.cat([alpha, sigmas], dim=1)
