import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class VisualBackbone(nn.Module):
    """
    Independent Visual Backbone based on EfficientNet-B0.
    Extracts 1280-dim features using Global Average Pooling.
    """

    def __init__(self):
        super().__init__()
        # Load EfficientNet-B0, pretrained on ImageNet
        # num_classes=0 removes the classification head
        # global_pool='avg' ensures we get the pooled feature vector (1280-dim)
        self.model = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

    def forward(self, x):
        # Input: (Batch, 3, 224, 224)
        # Output: (Batch, 1280)
        return self.model(x)


class TabularEncoder(nn.Module):
    """
    Encodes clinical metadata into a Shared Latent Vector.
    Architecture: Linear -> GeLU -> Linear -> GeLU
    """

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # We use a hidden dimension slightly larger than latent for capacity
        hidden_dim = latent_dim * 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # Input: (Batch, 7)
        # Output: (Batch, 128)
        return self.net(x)


class BBSLNet(nn.Module):
    """
    Balanced-Bottleneck Shared-Latent Network (BBSL-Net).
    Integrates dual-view CT scans with clinical metadata using a
    balanced fusion architecture to predict FVC trajectory parameters.
    """

    def __init__(self):
        super().__init__()

        # 1. Independent Visual Backbones
        self.backbone_ax = VisualBackbone()
        self.backbone_cor = VisualBackbone()

        # 2. Tabular Encoder
        # Input features: [Age, Sex, Smoke0, Smoke1, Smoke2, BaseFVC, BasePct]
        # We exclude 'Weeks' from the input to ensure static parameter prediction.
        self.tab_input_dim = 7
        self.latent_dim = Config.LATENT_DIM
        self.tab_encoder = TabularEncoder(self.tab_input_dim, self.latent_dim)

        # 3. Fusion / Alignment
        # Project latent (128) to backbone dim (1280) for attention
        self.align_proj = nn.Linear(self.latent_dim, Config.BACKBONE_DIM)
        # LayerNorm is critical here to match the scale of pretrained visual features
        self.align_ln = nn.LayerNorm(Config.BACKBONE_DIM)

        # 4. Pre-Norm Symmetric Attention
        # Transformer Encoder Layer allowing cross-modal interaction
        self.attention = nn.TransformerEncoderLayer(
            d_model=Config.BACKBONE_DIM,
            nhead=4,
            dim_feedforward=2048,
            dropout=Config.DROPOUT_RATE,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-Norm for stability
        )

        # 5. Balanced-Bottleneck Head
        # Compress the fused high-dim vector back to latent dim
        self.compress = nn.Linear(Config.BACKBONE_DIM, self.latent_dim)

        # Final Prediction Head
        # Input: Compressed Context (128) + Shared Latent (128) = 256
        # This enforces 50/50 contribution from Visual and Clinical streams
        self.head = nn.Linear(self.latent_dim * 2, 3)

    def forward(self, img_ax, img_cor, tabular):
        """
        Args:
            img_ax: (B, 3, 224, 224) - Axial Tri-Slab
            img_cor: (B, 3, 224, 224) - Coronal Tri-Slab
            tabular: (B, 8) - [Weeks, Age, Sex, Smoke0, Smoke1, Smoke2, BaseFVC, BasePct]

        Returns:
            alpha: (B,) - Slope of decline
            sigma_base: (B,) - Baseline uncertainty
            sigma_growth: (B,) - Uncertainty growth rate
        """
        # 1. Extract Visual Features
        v_ax = self.backbone_ax(img_ax)  # (B, 1280)
        v_cor = self.backbone_cor(img_cor)  # (B, 1280)

        # 2. Extract Tabular Latent
        # Strictly exclude 'Weeks' (index 0) to prevent leakage of time into static parameters
        tab_feats = tabular[:, 1:]  # (B, 7)
        t_lat = self.tab_encoder(tab_feats)  # (B, 128)

        # 3. Alignment Flow
        t_align = self.align_proj(t_lat)  # (B, 1280)
        t_align = self.align_ln(t_align)  # Normalize

        # 4. Contextualization
        # Stack tokens: [Axial, Coronal, Aligned_Tabular]
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)  # (B, 3, 1280)

        # Apply Self-Attention
        tokens = self.attention(tokens)  # (B, 3, 1280)

        # Holistic Readout (Global Average Pooling across tokens)
        h_fused = tokens.mean(dim=1)  # (B, 1280)

        # 5. Balanced-Bottleneck
        h_comp = self.compress(h_fused)  # (B, 128)

        # Concatenate with original Shared Latent (Prior Preservation)
        h_final = torch.cat([h_comp, t_lat], dim=1)  # (B, 256)

        # 6. Prediction
        out = self.head(h_final)  # (B, 3)

        # Unpack and Activate
        alpha = out[:, 0]
        sigma_base = F.softplus(out[:, 1])
        sigma_growth = F.softplus(out[:, 2])

        return alpha, sigma_base, sigma_growth
