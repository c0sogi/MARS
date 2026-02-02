import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class TabularGatedDualViewNetwork(nn.Module):
    """
    Tabular-Gated Dual-View Network for Lung Function Decline Prediction.

    Architecture:
    1. Independent Visual Backbones (EfficientNet-B0) for Axial and Coronal views.
    2. Up-Projected Tabular Embedding.
    3. Symmetric Attention Fusion (Visual + Tabular context).
    4. Tabular-Gated View Pooling (Dynamic aggregation based on clinical priors).
    5. Prior-Anchored Regression Head (Parametric inference).
    """

    def __init__(self):
        super(TabularGatedDualViewNetwork, self).__init__()

        # ==========================================
        # 1. Independent Visual Backbones
        # ==========================================
        # We use EfficientNet-B0 initialized with ImageNet weights.
        # num_classes=0 and global_pool='avg' ensures we get the 1280-dim feature vector directly.
        self.axial_backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        self.coronal_backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # ==========================================
        # 2. Up-Projected Tabular Embedding
        # ==========================================
        # Input dim is 7 (Age, Percent, Sex(2), Smoking(3))
        self.tab_input_dim = 7
        self.embed_dim = Config.BACKBONE_DIM  # 1280

        self.tabular_encoder = nn.Sequential(
            nn.Linear(self.tab_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(),
        )

        # ==========================================
        # 3. Symmetric Attention Fusion
        # ==========================================
        # Fuses [Axial, Coronal, Tabular] tokens
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=Config.ATTN_HEADS,
            dropout=Config.DROPOUT,
            batch_first=True,
        )

        # ==========================================
        # 4. Tabular-Gated View Pooling
        # ==========================================
        # "Clinical Director": Uses raw tabular features to weight the visual views.
        self.gating_mlp = nn.Sequential(
            nn.Linear(self.tab_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Outputs weights for [Axial, Coronal]
        )

        # ==========================================
        # 5. Prior-Anchored Head
        # ==========================================
        # Concatenates fused visual vector with raw tabular features.
        self.head_input_dim = self.embed_dim + self.tab_input_dim

        self.regressor = nn.Sequential(
            nn.Linear(self.head_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3),  # Outputs: alpha, sigma_base, sigma_growth
        )

    def forward(self, image_axial, image_coronal, tabular, dt, baseline_fvc):
        """
        Args:
            image_axial: (B, 3, 224, 224)
            image_coronal: (B, 3, 224, 224)
            tabular: (B, 7) Normalized clinical features
            dt: (B,) Time delta (weeks from baseline)
            baseline_fvc: (B,) Baseline FVC measurement

        Returns:
            dict containing predictions and intermediate parameters.
        """
        # ------------------------------------------
        # 1. Feature Extraction
        # ------------------------------------------
        # Extract global descriptors (B, 1280)
        feat_ax = self.axial_backbone(image_axial)
        feat_cor = self.coronal_backbone(image_coronal)

        # Embed tabular features (B, 1280)
        feat_tab = self.tabular_encoder(tabular)

        # ------------------------------------------
        # 2. Attention Fusion
        # ------------------------------------------
        # Stack tokens: [Axial, Coronal, Tabular] -> (B, 3, 1280)
        tokens = torch.stack([feat_ax, feat_cor, feat_tab], dim=1)

        # Self-Attention (Symmetric contextualization)
        # attn_output: (B, 3, 1280)
        attn_output, _ = self.attention(tokens, tokens, tokens)

        # Extract contextualized visual tokens
        ctx_ax = attn_output[:, 0, :]
        ctx_cor = attn_output[:, 1, :]

        # ------------------------------------------
        # 3. Gated Aggregation
        # ------------------------------------------
        # Predict weights based on clinical priors (B, 2)
        gating_logits = self.gating_mlp(tabular)
        gating_weights = F.softmax(gating_logits, dim=1)

        w_ax = gating_weights[:, 0].unsqueeze(1)  # (B, 1)
        w_cor = gating_weights[:, 1].unsqueeze(1)  # (B, 1)

        # Weighted sum of contextualized visual features
        feat_vis = (w_ax * ctx_ax) + (w_cor * ctx_cor)  # (B, 1280)

        # ------------------------------------------
        # 4. Prediction Head
        # ------------------------------------------
        # Skip connection: Concatenate fused visual with raw tabular
        head_input = torch.cat([feat_vis, tabular], dim=1)  # (B, 1287)

        # Predict parameters
        params = self.regressor(head_input)

        # Extract components
        alpha = params[:, 0]  # Slope can be negative or positive

        # Enforce positivity for uncertainty using Softplus
        sigma_base = F.softplus(params[:, 1])
        sigma_growth = F.softplus(params[:, 2])

        # ------------------------------------------
        # 5. Parametric Inference
        # ------------------------------------------
        # FVC = Baseline + alpha * dt
        fvc_pred = baseline_fvc + (alpha * dt)

        # Confidence = sigma_base + sigma_growth * |dt|
        confidence_pred = sigma_base + (sigma_growth * torch.abs(dt))

        return {
            "fvc_pred": fvc_pred,  # (B,)
            "confidence_pred": confidence_pred,  # (B,)
            "alpha": alpha,
            "sigma_base": sigma_base,
            "sigma_growth": sigma_growth,
            "gating_weights": gating_weights,  # Optional: for interpretability
        }
