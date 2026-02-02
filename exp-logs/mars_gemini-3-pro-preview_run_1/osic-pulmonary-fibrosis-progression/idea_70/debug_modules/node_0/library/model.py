import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class VisualBackbone(nn.Module):
    """
    Independent Low-Capacity Visual Backbone (EfficientNet-B0).
    Extracts high-fidelity features (1280-dim) without projection.
    """

    def __init__(self):
        super(VisualBackbone, self).__init__()
        # Load pretrained EfficientNet-B0
        # num_classes=0 ensures we get the Global Average Pooled features (1280-dim)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=Config.BACKBONE_PRETRAINED, num_classes=0
        )

    def forward(self, x):
        # Input: (B, 3, 224, 224)
        # Output: (B, 1280)
        return self.backbone(x)


class TabularEncoder(nn.Module):
    """
    Shared-Latent Tabular Encoder.
    Projects raw metadata to a robust Shared Latent Vector (T_lat).
    """

    def __init__(self):
        super(TabularEncoder, self).__init__()
        # Input features: Age, Sex, Smoking, Percent (4 features)
        input_dim = len(Config.TABULAR_FEATURES)
        hidden_dim = 64
        output_dim = Config.LATENT_DIM

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # Input: (B, 4)
        # Output: (B, 128)
        return self.mlp(x)


class PriorGatedAggregation(nn.Module):
    """
    Prior-Gated Aggregation Mechanism.
    1. Aligns tabular latent to visual dimension.
    2. Contextualizes views and prior using Pre-Norm Self-Attention.
    3. Dynamically weights visual tokens based on tabular context.
    4. Compresses result via Balanced Bottleneck.
    """

    def __init__(self):
        super(PriorGatedAggregation, self).__init__()

        self.vis_dim = Config.BACKBONE_OUT_DIM  # 1280
        self.lat_dim = Config.LATENT_DIM  # 128
        self.bottle_dim = Config.BOTTLENECK_DIM  # 128

        # Flow A: Fusion Alignment
        # Project T_lat (128) -> T_align (1280) + LayerNorm
        self.proj_align = nn.Linear(self.lat_dim, self.vis_dim)
        self.norm_align = nn.LayerNorm(self.vis_dim)

        # Contextualization: Pre-Norm Symmetric Attention
        # Single Transformer Encoder Layer
        self.attention_block = nn.TransformerEncoderLayer(
            d_model=self.vis_dim,
            nhead=8,  # 1280 / 8 = 160 per head
            dim_feedforward=2048,  # Standard expansion
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-Normalization
        )

        # Balanced Bottleneck
        # Input: Concat(WeightedVisual, ContextTabular) -> 1280 + 1280 = 2560
        # Output: 128
        self.bottleneck = nn.Sequential(
            nn.Linear(self.vis_dim * 2, self.bottle_dim), nn.GELU(), nn.Dropout(0.2)
        )

    def forward(self, v_ax, v_cor, t_lat):
        """
        Args:
            v_ax: Axial features (B, 1280)
            v_cor: Coronal features (B, 1280)
            t_lat: Tabular latent (B, 128)
        """
        batch_size = v_ax.size(0)

        # 1. Flow A: Align Tabular Prior
        t_align = self.proj_align(t_lat)  # (B, 1280)
        t_align = self.norm_align(t_align)  # Normalize to prevent shock

        # 2. Tokenization & Contextualization
        # Sequence: [Axial, Coronal, Tabular]
        seq = torch.stack([v_ax, v_cor, t_align], dim=1)  # (B, 3, 1280)

        # Transformer Pass
        seq_out = self.attention_block(seq)  # (B, 3, 1280)

        # Unpack tokens
        v_ax_prime = seq_out[:, 0, :]  # (B, 1280)
        v_cor_prime = seq_out[:, 1, :]  # (B, 1280)
        t_align_prime = seq_out[:, 2, :]  # (B, 1280)

        # 3. Prior-Gated Readout
        # Calculate attention weights: Softmax(T_align_prime . [V_ax_prime, V_cor_prime]^T)
        # Visual Keys: (B, 1280, 2)
        visual_keys = torch.stack([v_ax_prime, v_cor_prime], dim=2)

        # Query: t_align_prime (B, 1, 1280)
        query = t_align_prime.unsqueeze(1)

        # Scores: (B, 1, 2)
        scores = torch.bmm(query, visual_keys)
        attn_weights = F.softmax(scores, dim=2).squeeze(1)  # (B, 2)

        alpha_ax = attn_weights[:, 0].unsqueeze(1)  # (B, 1)
        alpha_cor = attn_weights[:, 1].unsqueeze(1)  # (B, 1)

        # Weighted Visual Context
        h_vis = alpha_ax * v_ax_prime + alpha_cor * v_cor_prime  # (B, 1280)

        # 4. Context Preservation
        # Concatenate Weighted Visual + Contextualized Tabular
        h_full = torch.cat([h_vis, t_align_prime], dim=1)  # (B, 2560)

        # 5. Balanced Bottleneck
        h_compressed = self.bottleneck(h_full)  # (B, 128)

        return h_compressed


class PGBBNet(nn.Module):
    """
    Prior-Gated Balanced-Bottleneck Network (PGBB-Net).
    """

    def __init__(self):
        super(PGBBNet, self).__init__()

        # 1. Independent Visual Backbones
        self.backbone_ax = VisualBackbone()
        self.backbone_cor = VisualBackbone()

        # 2. Shared-Latent Tabular Encoder
        self.tabular_encoder = TabularEncoder()

        # 3. Aggregation Mechanism
        self.aggregator = PriorGatedAggregation()

        # 4. Non-Linear Parametric Head
        # Input: Concat(h_compressed, t_lat_raw) -> 128 + 128 = 256
        self.head = nn.Sequential(
            nn.Linear(Config.BOTTLENECK_DIM + Config.LATENT_DIM, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # Outputs: alpha (slope), sigma_base, sigma_growth
        )

    def forward(self, axial, coronal, tabular, meta):
        """
        Args:
            axial: (B, 3, 224, 224)
            coronal: (B, 3, 224, 224)
            tabular: (B, 4) [Age, Sex, Smoke, Percent]
            meta: (B, 2) [rel_week, base_fvc]
        Returns:
            preds: (B, 2) [FVC_pred, Confidence_pred]
        """
        # 1. Feature Extraction
        v_ax = self.backbone_ax(axial)  # (B, 1280)
        v_cor = self.backbone_cor(coronal)  # (B, 1280)
        t_lat = self.tabular_encoder(tabular)  # (B, 128)

        # 2. Aggregation & Bottleneck
        h_compressed = self.aggregator(v_ax, v_cor, t_lat)  # (B, 128)

        # 3. Final Assembly (Dimensionality Balancing)
        # Concatenate Compressed Context (128) with Raw Prior (128)
        # This ensures the visual residual does not drown out the dominant clinical prior
        final_vec = torch.cat([h_compressed, t_lat], dim=1)  # (B, 256)

        # 4. Parametric Prediction
        params = self.head(final_vec)

        slope = params[:, 0]
        sigma_base = F.softplus(params[:, 1])
        sigma_growth = F.softplus(params[:, 2])

        # 5. Trajectory Calculation
        # meta contains: [rel_week, base_fvc]
        rel_week = meta[:, 0]
        base_fvc = meta[:, 1]

        # FVC = Baseline + Slope * DeltaT
        fvc_pred = base_fvc + slope * rel_week

        # Confidence = Base + Growth * |DeltaT|
        sigma_pred = sigma_base + sigma_growth * torch.abs(rel_week)

        return torch.stack([fvc_pred, sigma_pred], dim=1)
