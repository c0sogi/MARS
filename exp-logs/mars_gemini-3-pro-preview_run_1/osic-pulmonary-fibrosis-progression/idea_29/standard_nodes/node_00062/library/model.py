import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class CVRNet(nn.Module):
    """
    Contextualized-Visual-Residual Network (CVR-Net).

    Architecture:
    1. Independent High-Fidelity Backbones (EfficientNet-B0) for Axial and Coronal views.
    2. Up-Projected Tabular Embedding (MLP).
    3. Symmetric Attention with Pre-Norm (Contextualization).
    4. Visual-Exclusive Pooled Readout (Isolation).
    5. Prior-Anchored Head (Anchoring).
    """

    def __init__(self):
        super(CVRNet, self).__init__()

        # ==========================
        # 1. Visual Backbones
        # ==========================
        # Independent backbones for orthogonal views.
        # num_classes=0 returns the Global Average Pooled features (1280-dim for B0).
        self.backbone_axial = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, num_classes=0
        )

        self.backbone_coronal = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, num_classes=0
        )

        self.visual_dim = Config.HIDDEN_DIM  # 1280

        # ==========================
        # 2. Tabular Encoder
        # ==========================
        # Projects fusion features (Age, Sex, Smoke*3, Pct) UP to visual dim.
        # Input dim is 6 based on LungDataset fusion_list.
        self.fusion_input_dim = 6
        self.tabular_mlp = nn.Sequential(
            nn.Linear(self.fusion_input_dim, Config.TABULAR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.TABULAR_HIDDEN_DIM, self.visual_dim),
        )

        # ==========================
        # 3. Fusion Layer
        # ==========================
        # Pre-Norm Multi-Head Self-Attention.
        self.layer_norm = nn.LayerNorm(self.visual_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.visual_dim, num_heads=Config.FUSION_HEADS, batch_first=True
        )

        # ==========================
        # 4. Residual Head
        # ==========================
        # Input: Visual Residual (1280) + Anchor Features (3)
        # Anchor features: Baseline FVC (norm), Baseline Pct (norm), Week Diff (norm)
        self.anchor_dim = 3
        self.head_input_dim = self.visual_dim + self.anchor_dim

        self.head = nn.Sequential(
            nn.Linear(self.head_input_dim, 512),
            nn.ReLU(),
            nn.Linear(
                512, Config.OUTPUT_DIM
            ),  # Output: alpha, sigma_base, sigma_growth
        )

    def forward(self, axial, coronal, fusion, anchor):
        """
        Args:
            axial: (B, 3, 224, 224)
            coronal: (B, 3, 224, 224)
            fusion: (B, 6) - Context features
            anchor: (B, 3) - Prior features for the head

        Returns:
            alpha: (B,) - Slope of decline
            sigma_base: (B,) - Base uncertainty
            sigma_growth: (B,) - Uncertainty growth rate
        """
        batch_size = axial.size(0)

        # --- 1. Feature Extraction ---
        # Extract high-fidelity visual features (B, 1280)
        v_ax = self.backbone_axial(axial)
        v_cor = self.backbone_coronal(coronal)

        # Encode tabular context (B, 1280)
        v_tab = self.tabular_mlp(fusion)

        # --- 2. Contextualization (Fusion) ---
        # Stack tokens: [Axial, Coronal, Tabular] -> (B, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, v_tab], dim=1)

        # Pre-Norm
        tokens_norm = self.layer_norm(tokens)

        # Self-Attention
        attn_out, _ = self.attention(tokens_norm, tokens_norm, tokens_norm)

        # Residual Connection
        tokens = tokens + attn_out

        # --- 3. Isolation ---
        # Extract updated visual tokens (indices 0 and 1)
        v_ax_new = tokens[:, 0, :]
        v_cor_new = tokens[:, 1, :]

        # Average Pool to get Contextualized Visual Residual
        visual_residual = (v_ax_new + v_cor_new) / 2.0

        # --- 4. Anchoring ---
        # Concatenate with raw anchor features
        combined = torch.cat([visual_residual, anchor], dim=1)

        # --- 5. Prediction ---
        out = self.head(combined)

        # Split outputs
        alpha = out[:, 0]

        # Enforce positivity for sigmas using Softplus
        sigma_base = F.softplus(out[:, 1])
        sigma_growth = F.softplus(out[:, 2])

        return alpha, sigma_base, sigma_growth
