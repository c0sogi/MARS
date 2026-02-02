import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling layer.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeMPooling, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        # Clamp for numerical stability
        x = x.clamp(min=self.eps)
        # Average pooling on x^p
        # (B, C, H, W) -> (B, C, 1, 1)
        x = F.avg_pool2d(x.pow(self.p), (x.size(-2), x.size(-1)))
        # (x_avg)^(1/p)
        x = x.pow(1.0 / self.p)
        return x.flatten(1)


class DynamicDepthGeMNet(nn.Module):
    """
    Dual-Axis EfficientNet with GeM Pooling and Symmetric Attention Fusion.
    Predicts FVC decline parameters based on Axial/Coronal CT views and clinical metadata.
    """

    def __init__(self):
        super(DynamicDepthGeMNet, self).__init__()

        # 1. Visual Backbones (Independent)
        # EfficientNet-B0 output features are 1280 channels
        self.backbone_ax = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            out_indices=(4,),
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            out_indices=(4,),
        )

        # Feature dimension for EfficientNet-B0
        self.vis_dim = 320

        # 2. Pooling
        self.gem = GeMPooling(p=Config.GEM_P_INIT)

        # 3. Projections to Hidden Dimension
        self.proj_vis = nn.Linear(self.vis_dim, Config.HIDDEN_DIM)
        self.proj_tab = nn.Linear(len(Config.TABULAR_COLS), Config.HIDDEN_DIM)

        # 4. Attention Fusion
        # Tokens: [Axial, Coronal, Tabular]
        self.attn = nn.MultiheadAttention(
            embed_dim=Config.HIDDEN_DIM,
            num_heads=4,
            batch_first=True,
            dropout=Config.DROPOUT,
        )

        # 5. Regression Head
        # Input: Attended Tabular Token + Raw Tabular Features (Skip Connection)
        head_in_dim = Config.HIDDEN_DIM + len(Config.TABULAR_COLS)

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, Config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM // 2, 3),  # alpha, sigma_base, sigma_growth
        )

    def forward(self, inputs):
        """
        Args:
            inputs (dict): Contains:
                'img_ax': (B, 3, H, W)
                'img_cor': (B, 3, H, W)
                'tab': (B, 5) [week_norm, pct, age, sex, smoke]
                'base_fvc': (B, 1)
        Returns:
            fvc_pred: (B, 1)
            sigma_pred: (B, 1)
        """
        img_ax = inputs["img_ax"]
        img_cor = inputs["img_cor"]
        tab = inputs["tab"]
        base_fvc = inputs["base_fvc"]

        batch_size = img_ax.size(0)

        # --- 1. Visual Feature Extraction ---
        # Get last feature map (B, 1280, 7, 7)
        feat_ax = self.backbone_ax(img_ax)[0]
        feat_cor = self.backbone_cor(img_cor)[0]

        # --- 2. GeM Pooling ---
        # (B, 1280)
        emb_ax = self.gem(feat_ax)
        emb_cor = self.gem(feat_cor)

        # --- 3. Tokenization ---
        # Project to hidden dim (B, 512)
        tok_ax = self.proj_vis(emb_ax)
        tok_cor = self.proj_vis(emb_cor)
        tok_tab = self.proj_tab(tab)

        # Stack tokens: (B, 3, 512)
        tokens = torch.stack([tok_ax, tok_cor, tok_tab], dim=1)

        # --- 4. Attention Fusion ---
        # Self-attention allows modalities to contextulize each other
        attn_out, _ = self.attn(tokens, tokens, tokens)

        # Extract the refined tabular token (index 2)
        refined_tab = attn_out[:, 2, :]

        # --- 5. Prediction Head ---
        # Skip connection: Concatenate refined token with raw tabular input
        head_input = torch.cat([refined_tab, tab], dim=1)

        # Predict parameters: alpha, sigma_base, sigma_growth
        params = self.head(head_input)

        alpha = params[:, 0:1]
        sigma_base = params[:, 1:2]
        sigma_growth = params[:, 2:3]

        # Enforce positivity for sigmas
        sigma_base = F.softplus(sigma_base)
        sigma_growth = F.softplus(sigma_growth)

        # --- 6. Calculate FVC and Confidence ---
        # tab[:, 0] is normalized week: delta_week / 100.0
        # We need delta_week for calculation
        delta_week = tab[:, 0:1] * 100.0

        # FVC = Base + alpha * delta_week
        fvc_pred = base_fvc + alpha * delta_week

        # Confidence = sigma_base + sigma_growth * |delta_week|
        confidence_pred = sigma_base + sigma_growth * torch.abs(delta_week)

        return fvc_pred, confidence_pred
