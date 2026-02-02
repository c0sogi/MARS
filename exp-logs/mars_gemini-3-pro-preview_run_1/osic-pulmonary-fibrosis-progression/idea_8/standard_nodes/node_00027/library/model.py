import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class TabularAnchor(nn.Module):
    """
    Processes clinical metadata to produce:
    1. Baseline trajectory parameters (alpha, sigma_base, sigma_growth).
    2. A query vector for the visual attention mechanism.
    """

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(Config.TABULAR_INPUT_DIM, Config.TABULAR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.TABULAR_HIDDEN_DIM, Config.TABULAR_HIDDEN_DIM),
            nn.ReLU(),
        )

        # Output 1: Baseline parameters [alpha, sigma_base, sigma_growth]
        self.head_params = nn.Linear(Config.TABULAR_HIDDEN_DIM, 3)

        # Output 2: Query vector for attention
        self.head_query = nn.Linear(Config.TABULAR_HIDDEN_DIM, Config.ATTENTION_DIM)

    def forward(self, x):
        feat = self.mlp(x)
        params = self.head_params(feat)
        query = self.head_query(feat)
        return params, query


class DualVisualBackbone(nn.Module):
    """
    Two independent EfficientNet-B0 backbones for Axial and Coronal views.
    Extracts global features and projects them to the attention dimension.
    """

    def __init__(self):
        super().__init__()
        # Load pretrained EfficientNet-B0
        # num_classes=0 with global_pool='' (default in timm for num_classes=0 usually means pool)
        # We want the global feature vector.
        self.backbone_ax = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            in_chans=Config.IN_CHANNELS,
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            in_chans=Config.IN_CHANNELS,
        )

        # Determine feature dimension (1280 for EfficientNet-B0)
        self.feat_dim = self.backbone_ax.num_features

        # Projection layer to match Attention Dimension
        self.proj = nn.Linear(self.feat_dim, Config.ATTENTION_DIM)

    def forward(self, img_ax, img_cor):
        # Extract features (Batch, Feat_Dim)
        f_ax = self.backbone_ax(img_ax)
        f_cor = self.backbone_cor(img_cor)

        # Project to Attention Dimension (Batch, Attn_Dim)
        k_ax = self.proj(f_ax)
        k_cor = self.proj(f_cor)

        return k_ax, k_cor


class CrossAttentionModule(nn.Module):
    """
    Cross-Attention mechanism where Tabular Query attends to Visual Keys.
    """

    def __init__(self):
        super().__init__()
        self.scale = Config.ATTENTION_DIM**-0.5

    def forward(self, query, key_ax, key_cor):
        """
        Args:
            query: (B, Dim) - From Tabular
            key_ax: (B, Dim) - From Axial View
            key_cor: (B, Dim) - From Coronal View
        Returns:
            context: (B, Dim) - Weighted combination of visual features
        """
        # Prepare Query: (B, 1, Dim)
        Q = query.unsqueeze(1)

        # Prepare Keys/Values: Stack visual features -> (B, 2, Dim)
        K = torch.stack([key_ax, key_cor], dim=1)
        V = K  # We use the same features for Keys and Values

        # Calculate Attention Scores
        # (B, 1, Dim) @ (B, Dim, 2) -> (B, 1, 2)
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        weights = F.softmax(scores, dim=-1)

        # Compute Context
        # (B, 1, 2) @ (B, 2, Dim) -> (B, 1, Dim)
        context = torch.bmm(weights, V).squeeze(1)

        return context


class ResidualCrossAttentionNet(nn.Module):
    """
    Main Architecture:
    1. Tabular Anchor -> Baseline Params + Query
    2. Visual Backbones -> Visual Features
    3. Cross Attention -> Visual Context
    4. Correction Head -> Param Deltas
    5. Final Prediction -> FVC, Confidence
    """

    def __init__(self):
        super().__init__()
        self.tabular_anchor = TabularAnchor()
        self.visual_backbone = DualVisualBackbone()
        self.attention = CrossAttentionModule()

        # Correction Head: Predicts residuals for [alpha, sigma_base, sigma_growth]
        self.correction_head = nn.Sequential(
            nn.Linear(Config.ATTENTION_DIM, Config.TABULAR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.TABULAR_HIDDEN_DIM, 3),
        )

    def forward(self, tabular, img_ax, img_cor, relative_week, baseline_fvc):
        """
        Args:
            tabular: (B, 7)
            img_ax: (B, 3, 224, 224)
            img_cor: (B, 3, 224, 224)
            relative_week: (B,) or (B, 1)
            baseline_fvc: (B,) or (B, 1)
        """
        # 1. Tabular Pathway
        params_base, query = self.tabular_anchor(tabular)

        # 2. Visual Pathway
        k_ax, k_cor = self.visual_backbone(img_ax, img_cor)

        # 3. Cross Attention
        context = self.attention(query, k_ax, k_cor)

        # 4. Residual Correction
        delta_params = self.correction_head(context)

        # 5. Combine Parameters
        # params: [alpha, sigma_base, sigma_growth]
        params_final = params_base + delta_params

        alpha = params_final[:, 0]
        sigma_base = params_final[:, 1]
        sigma_growth = params_final[:, 2]

        # 6. Apply Constraints
        # Alpha (slope) is unconstrained (can be negative for decline)
        # Sigmas must be positive -> Softplus
        sigma_base = F.softplus(sigma_base)
        sigma_growth = F.softplus(sigma_growth)

        # 7. Parametric Inference
        # Ensure inputs are correct shape
        if relative_week.dim() == 1:
            relative_week = relative_week.view(-1)
        if baseline_fvc.dim() == 1:
            baseline_fvc = baseline_fvc.view(-1)

        # FVC = Baseline + alpha * week
        fvc_pred = baseline_fvc + alpha * relative_week

        # Confidence = sigma_base + sigma_growth * |week|
        confidence_pred = sigma_base + sigma_growth * torch.abs(relative_week)

        return fvc_pred, confidence_pred
