import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class VisualBackbone(nn.Module):
    """
    Independent Low-Capacity Visual Backbone based on EfficientNet-B0.
    Extracts high-fidelity features without dimensionality compression.
    """

    def __init__(self, pretrained=True):
        super(VisualBackbone, self).__init__()
        # Load EfficientNet-B0
        # num_classes=0 removes the classifier, global_pool='' returns feature maps
        # We will handle pooling manually to be explicit
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",  # Returns (Batch, 1280) directly
        )

    def forward(self, x):
        # x: (Batch, 3, 224, 224)
        # output: (Batch, 1280)
        return self.backbone(x)


class SharedLatentEncoder(nn.Module):
    """
    Processes raw clinical metadata into a robust Shared Latent Vector (T_lat).
    """

    def __init__(self, input_dim=6, latent_dim=128):
        super(SharedLatentEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: (Batch, 6)
        # output: (Batch, 128)
        return self.net(x)


class FusionBlock(nn.Module):
    """
    Normalized Bifurcated Flow & Pre-Norm Symmetric Attention.
    Fuses visual tokens with the aligned clinical latent vector.
    """

    def __init__(self, embed_dim=1280, latent_dim=128, num_heads=8, dropout=0.1):
        super(FusionBlock, self).__init__()

        # Flow A: Alignment Projection + LayerNorm
        self.align_proj = nn.Linear(latent_dim, embed_dim)
        self.align_norm = nn.LayerNorm(embed_dim)

        # Attention Mechanism (Pre-Norm)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout
        )

        # Feed Forward Network (Pre-Norm)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, v_ax, v_cor, t_lat):
        """
        Args:
            v_ax: Axial visual vector (Batch, 1280)
            v_cor: Coronal visual vector (Batch, 1280)
            t_lat: Shared latent tabular vector (Batch, 128)
        """
        # 1. Flow A: Project and Normalize Tabular Latent
        t_align = self.align_proj(t_lat)
        t_align = self.align_norm(t_align)  # (Batch, 1280)

        # 2. Tokenization: Stack [Axial, Coronal, Aligned_Tabular]
        # Shape: (Batch, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)

        # 3. Pre-Norm Self-Attention
        # Residual connection 1
        x_norm1 = self.norm1(tokens)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = tokens + attn_out

        # Residual connection 2 (FFN)
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + ffn_out

        # 4. Holistic Readout: Global Average Pooling across tokens
        # Shape: (Batch, 1280)
        h_fused = torch.mean(x, dim=1)

        return h_fused


class PriorAnchoredHead(nn.Module):
    """
    Bottleneck Prior-Anchored Head.
    Combines the fused holistic context with the original shared latent prior
    to predict trajectory parameters.
    """

    def __init__(self, fused_dim=1280, latent_dim=128, dropout=0.1):
        super(PriorAnchoredHead, self).__init__()

        input_dim = fused_dim + latent_dim  # 1408

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2), nn.GELU(), nn.Dropout(dropout)
        )

        # Prediction Heads
        # We predict 3 parameters: alpha (slope), sigma_base, sigma_growth
        self.head = nn.Linear(input_dim // 2, 3)

    def forward(self, h_fused, t_lat):
        # Concatenate Flow B (Prior Preservation)
        combined = torch.cat([h_fused, t_lat], dim=1)  # (Batch, 1408)

        feat = self.bottleneck(combined)
        out = self.head(feat)

        # Split outputs
        alpha = out[:, 0]  # Slope (can be negative or positive)
        sigma_base = out[:, 1]  # Base confidence
        sigma_growth = out[:, 2]  # Confidence growth rate

        # Apply constraints
        # Sigmas must be positive. Softplus is smooth and ensures positivity.
        sigma_base = F.softplus(sigma_base)
        sigma_growth = F.softplus(sigma_growth)

        return alpha, sigma_base, sigma_growth


class NSLHN(nn.Module):
    """
    Normalized Shared-Latent Holistic Network (NSL-HN).

    Architecture:
    1. Independent EfficientNet-B0 backbones for Axial and Coronal views.
    2. Shared Latent MLP for tabular data.
    3. Normalized Fusion Block (Attention).
    4. Prior-Anchored Head for parametric trajectory prediction.
    """

    def __init__(self):
        super(NSLHN, self).__init__()

        # 1. Visual Backbones
        self.axial_backbone = VisualBackbone(pretrained=Config.BACKBONE_PRETRAINED)
        self.coronal_backbone = VisualBackbone(pretrained=Config.BACKBONE_PRETRAINED)

        # 2. Tabular Encoder
        self.tabular_encoder = SharedLatentEncoder(
            input_dim=6, latent_dim=Config.LATENT_DIM
        )

        # 3. Fusion Block
        self.fusion_block = FusionBlock(
            embed_dim=Config.BACKBONE_OUT_DIM,
            latent_dim=Config.LATENT_DIM,
            dropout=Config.DROPOUT_RATE,
        )

        # 4. Prediction Head
        self.head = PriorAnchoredHead(
            fused_dim=Config.FUSED_DIM,
            latent_dim=Config.LATENT_DIM,
            dropout=Config.DROPOUT_RATE,
        )

    def forward(self, axial, coronal, tabular, base_fvc, delta_week):
        """
        Args:
            axial: (Batch, 3, 224, 224)
            coronal: (Batch, 3, 224, 224)
            tabular: (Batch, 6)
            base_fvc: (Batch,) - The baseline FVC measurement
            delta_week: (Batch,) - The time difference (Predict_Week - Base_Week)

        Returns:
            pred_fvc: (Batch,) - Predicted FVC
            pred_sigma: (Batch,) - Predicted Confidence
        """
        # 1. Extract Features
        v_ax = self.axial_backbone(axial)  # (Batch, 1280)
        v_cor = self.coronal_backbone(coronal)  # (Batch, 1280)
        t_lat = self.tabular_encoder(tabular)  # (Batch, 128)

        # 2. Fuse
        h_fused = self.fusion_block(v_ax, v_cor, t_lat)  # (Batch, 1280)

        # 3. Predict Parameters
        alpha, sigma_base, sigma_growth = self.head(h_fused, t_lat)

        # 4. Calculate Trajectory (Parametric Inference)
        # FVC = Base + alpha * delta_t
        pred_fvc = base_fvc + alpha * delta_week

        # Sigma = Base_Sigma + Growth_Sigma * |delta_t|
        pred_sigma = sigma_base + sigma_growth * torch.abs(delta_week)

        return pred_fvc, pred_sigma
