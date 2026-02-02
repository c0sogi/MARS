import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class IdentityFusion(nn.Module):
    """
    Identity-Injected Symmetric Fusion Module.

    Injects learnable identity embeddings into the feature sequence to allow
    the Transformer to distinguish between modalities (Axial, Coronal, Tabular)
    despite the permutation invariance of self-attention.
    Uses a Pre-Norm Transformer Encoder for stability.
    """

    def __init__(self, feature_dim, num_heads=4, ff_dim=2048, dropout=0.1):
        super().__init__()

        # Learnable identity embeddings for the three modalities
        # 0: Axial, 1: Coronal, 2: Tabular
        self.identity_embeddings = nn.Parameter(torch.randn(1, 3, feature_dim) * 0.02)

        # Pre-Norm Transformer Encoder Layer
        # norm_first=True implements Pre-Normalization (LayerNorm -> Attention -> Add)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        """
        Args:
            x: Stacked features [Batch, 3, Feature_Dim]
               Order must be [Axial, Coronal, Tabular]
        """
        # Add identity embeddings (broadcasting across batch)
        x = x + self.identity_embeddings

        # Pass through Transformer
        x = self.transformer(x)

        return x


class IASDANet(nn.Module):
    """
    Identity-Aware Symmetric Dual-Axis Network (IAS-DAN).

    Architecture:
    1. Independent EfficientNet-B0 backbones for Axial and Coronal views.
    2. Up-Projected MLP for tabular metadata.
    3. Identity-Injected Symmetric Fusion (Transformer).
    4. Global Average Pooling over modalities.
    5. Prior-Anchored Regression Head predicting trajectory parameters.
    """

    def __init__(self):
        super().__init__()

        # ==========================================
        # 1. Independent Visual Backbones
        # ==========================================
        # We use num_classes=0 to get the pooled feature vector directly (1280 dim for B0)
        self.backbone_ax = timm.create_model(
            Config.BACKBONE, pretrained=Config.BACKBONE_PRETRAINED, num_classes=0
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE, pretrained=Config.BACKBONE_PRETRAINED, num_classes=0
        )

        # Feature dimension for EfficientNet-B0 is 1280
        self.feature_dim = self.backbone_ax.num_features  # 1280

        # ==========================================
        # 2. Tabular Projection
        # ==========================================
        # Input: Age, Sex, Smoking(3), Percent -> 6 dims
        input_tab_dim = 6
        self.tabular_mlp = nn.Sequential(
            nn.Linear(input_tab_dim, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(Config.HIDDEN_DIM, self.feature_dim),  # Up-project to 1280
        )

        # ==========================================
        # 3. Identity-Injected Fusion
        # ==========================================
        self.fusion = IdentityFusion(
            feature_dim=self.feature_dim, num_heads=8, ff_dim=2048, dropout=0.1
        )

        # ==========================================
        # 4. Trajectory Head
        # ==========================================
        # Input: Pooled Context (1280) + Raw Tabular Skip (6)
        head_input_dim = self.feature_dim + input_tab_dim

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(Config.HIDDEN_DIM, 3),  # alpha, sigma_base, sigma_growth
        )

    def forward(
        self, axial_img, coronal_img, tabular, time_delta, baseline_fvc, **kwargs
    ):
        """
        Args:
            axial_img: [B, 3, 224, 224]
            coronal_img: [B, 3, 224, 224]
            tabular: [B, 6]
            time_delta: [B, 1] (Week - Baseline_Week)
            baseline_fvc: [B, 1]

        Returns:
            fvc_pred: [B, 1]
            sigma_pred: [B, 1]
        """
        # 1. Visual Feature Extraction
        # Backbones expect [B, 3, H, W]
        # Output: [B, 1280]
        feat_ax = self.backbone_ax(axial_img)
        feat_cor = self.backbone_cor(coronal_img)

        # 2. Tabular Feature Extraction
        # Output: [B, 1280]
        feat_tab = self.tabular_mlp(tabular)

        # 3. Stack Modalities
        # Sequence: [Axial, Coronal, Tabular] -> [B, 3, 1280]
        sequence = torch.stack([feat_ax, feat_cor, feat_tab], dim=1)

        # 4. Identity-Injected Fusion
        # Output: [B, 3, 1280]
        context_seq = self.fusion(sequence)

        # 5. Holistic Pooling
        # Global Average Pooling across the sequence dimension
        # Output: [B, 1280]
        context_vector = torch.mean(context_seq, dim=1)

        # 6. Prior-Anchored Head
        # Concatenate context with raw tabular features (Skip Connection)
        # Output: [B, 1286]
        combined = torch.cat([context_vector, tabular], dim=1)

        # Predict parameters
        # out: [B, 3] -> alpha, sigma_base, sigma_growth
        out = self.head(combined)

        alpha = out[:, 0:1]  # Slope (can be negative)
        sigma_base = out[:, 1:2]  # Base uncertainty
        sigma_growth = out[:, 2:3]  # Growth uncertainty

        # Enforce positivity for sigmas using Softplus
        sigma_base = F.softplus(sigma_base)
        sigma_growth = F.softplus(sigma_growth)

        # 7. Parametric Inference
        # FVC = Baseline + alpha * delta_t
        fvc_pred = baseline_fvc + alpha * time_delta

        # Confidence = sigma_base + sigma_growth * |delta_t|
        sigma_pred = sigma_base + sigma_growth * torch.abs(time_delta)

        return fvc_pred, sigma_pred
