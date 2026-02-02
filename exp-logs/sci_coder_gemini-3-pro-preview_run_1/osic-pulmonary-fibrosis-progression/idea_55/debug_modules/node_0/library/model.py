import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class TabularEncoder(nn.Module):
    """
    Deep MLP to encode raw clinical features into a Shared Latent Vector.
    Structure: Linear -> GeLU -> Linear -> GeLU
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class PreNormAttention(nn.Module):
    """
    Multi-Head Self-Attention block with Pre-Normalization.
    Used to fuse visual and tabular tokens without gradient explosion.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Embed_Dim)

        # 1. Self-Attention Block (Pre-Norm)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # 2. Feed-Forward Block (Pre-Norm)
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x


class NSLHN(nn.Module):
    """
    Normalized Shared-Latent Holistic Network (NSL-HN).

    Integrates:
    1. Independent Low-Capacity Visual Backbones (EfficientNet-B0)
    2. Shared-Latent Tabular Encoder
    3. Normalized Bifurcated Flow for stability
    4. Pre-Norm Symmetric Attention for fusion
    5. Prior-Anchored Head for parametric prediction
    """

    def __init__(self):
        super().__init__()

        # ==========================================
        # 1. Independent Visual Backbones
        # ==========================================
        # We use num_classes=0 to get the Global Average Pooled features (1280-dim)
        # without the final classification layer.
        self.backbone_ax = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0
        )

        # Native output dimension of EfficientNet-B0
        self.visual_dim = Config.VISUAL_DIM

        # ==========================================
        # 2. Shared-Latent Tabular Encoder
        # ==========================================
        # Input dim is 6 (Age, Sex, 3xSmoking, Percent)
        self.tabular_encoder = TabularEncoder(
            input_dim=6, hidden_dim=64, output_dim=Config.LATENT_DIM
        )

        # ==========================================
        # 3. Normalized Bifurcated Flow
        # ==========================================
        # Flow A: Project Latent to Visual Dim for Alignment
        self.align_proj = nn.Linear(Config.LATENT_DIM, self.visual_dim)
        # Critical: LayerNorm immediately after projection to match visual backbone statistics
        self.align_norm = nn.LayerNorm(self.visual_dim)

        # ==========================================
        # 4. Pre-Norm Symmetric Attention
        # ==========================================
        self.attention = PreNormAttention(
            embed_dim=self.visual_dim, num_heads=4, dropout=Config.DROPOUT_RATE
        )

        # ==========================================
        # 5. Bottleneck Prior-Anchored Head
        # ==========================================
        # Concatenate Holistic Context (1280) + Original Latent (128)
        combined_dim = self.visual_dim + Config.LATENT_DIM

        self.bottleneck = nn.Sequential(
            nn.Linear(combined_dim, Config.HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        # Output: Alpha (Slope), Sigma_Base, Sigma_Growth
        self.head = nn.Linear(Config.HEAD_HIDDEN_DIM, 3)

    def forward(self, axial_img, coronal_img, tabular):
        """
        Args:
            axial_img: (B, 3, 224, 224)
            coronal_img: (B, 3, 224, 224)
            tabular: (B, 6)

        Returns:
            alpha: (B,) - Slope of decline
            sigma_base: (B,) - Base uncertainty
            sigma_growth: (B,) - Uncertainty growth rate
        """
        # 1. Visual Extraction
        # Output shape: (B, 1280)
        v_ax = self.backbone_ax(axial_img)
        v_cor = self.backbone_cor(coronal_img)

        # 2. Tabular Encoding
        # Output shape: (B, 128)
        t_lat = self.tabular_encoder(tabular)

        # 3. Normalization Flow (Alignment)
        # Project and Normalize: (B, 128) -> (B, 1280)
        t_align = self.align_proj(t_lat)
        t_align = self.align_norm(t_align)

        # 4. Attention Fusion
        # Stack tokens: [Axial, Coronal, Aligned_Tabular]
        # Shape: (B, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)

        # Contextualize via Self-Attention
        tokens_out = self.attention(tokens)

        # Holistic Readout: Global Average Pooling over sequence dim
        # Shape: (B, 1280)
        h_fused = torch.mean(tokens_out, dim=1)

        # 5. Prediction Head
        # Concatenate Fused Context with Original Latent (Prior Preservation)
        # Shape: (B, 1408)
        combined = torch.cat([h_fused, t_lat], dim=1)

        # Bottleneck
        feat = self.bottleneck(combined)

        # Final Projection
        out = self.head(feat)

        # Extract Parameters
        alpha = out[:, 0]  # Linear slope (unconstrained)
        sigma_base = F.softplus(out[:, 1])  # Must be positive
        sigma_growth = F.softplus(out[:, 2])  # Must be positive

        return alpha, sigma_base, sigma_growth
