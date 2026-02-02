import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Formula: f = (1/|X| * sum(x^p))^(1/p)

    Learns the pooling parameter p to interpolate between Average Pooling (p=1)
    and Max Pooling (p -> infinity). This is beneficial for capturing focal
    fibrosis signals in lung CT scans.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp to avoid numerical instability with power operations
        x = torch.clamp(x, min=self.eps)

        # Average pooling over spatial dimensions (H, W) on x^p
        # This computes (1/HW * sum(x^p))
        x_pow = x.pow(self.p)
        avg_pool = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))

        # Raise to power 1/p
        return avg_pool.pow(1.0 / self.p)


class DualAxisTransformer(nn.Module):
    """
    GeM-Pooled Dual-Axis Transformer for Lung Function Decline Prediction.

    Components:
    1. Two independent EfficientNet-B0 backbones (Axial & Coronal).
    2. GeM Pooling for robust feature aggregation.
    3. Tabular MLP for clinical metadata embedding.
    4. Transformer Encoder for multi-modal fusion.
    5. Residual Anchor (Skip Connection) for tabular features.
    6. Parametric Prediction Head (Alpha, Sigma_Base, Sigma_Growth).
    """

    def __init__(self):
        super(DualAxisTransformer, self).__init__()

        # ==========================================
        # 1. Independent Visual Backbones
        # ==========================================
        # We use features_only=False, num_classes=0, global_pool='' to get raw feature maps
        # EfficientNet-B0 output channels = 1280
        self.backbone_ax = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, num_classes=0, global_pool=""
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, num_classes=0, global_pool=""
        )

        # Feature dimension for EfficientNet-B0
        self.vis_dim = 1280

        # ==========================================
        # 2. Learnable GeM Pooling
        # ==========================================
        self.gem_ax = GeM(p=Config.GEM_P_INIT)
        self.gem_cor = GeM(p=Config.GEM_P_INIT)

        # ==========================================
        # 3. Projections
        # ==========================================
        # Project visual features to Token Dimension
        self.vis_proj = nn.Linear(self.vis_dim, Config.TOKEN_DIM)

        # ==========================================
        # 4. Tabular Embedding
        # ==========================================
        # Input features: Age, Percent, Base_FVC, Sex(2), Smoke(3) -> Total 8
        self.tab_input_dim = 8
        self.tab_mlp = nn.Sequential(
            nn.Linear(self.tab_input_dim, Config.TOKEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(Config.TOKEN_DIM // 2, Config.TOKEN_DIM),
        )

        # ==========================================
        # 5. Transformer Fusion
        # ==========================================
        # Sequence: [Axial_Token, Coronal_Token, Tabular_Token]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.TOKEN_DIM,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=Config.TOKEN_DIM * 4,
            dropout=Config.DROPOUT,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        # ==========================================
        # 6. Prediction Head with Residual Anchor
        # ==========================================
        # Input: Concatenation of (Transformer_Output_Tabular, Original_Tabular_Token)
        self.head_input_dim = Config.TOKEN_DIM * 2

        self.head = nn.Sequential(
            nn.Linear(self.head_input_dim, Config.TOKEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.TOKEN_DIM, 3),  # Outputs: Alpha, Sigma_Base, Sigma_Growth
        )

    def forward(self, img_ax, img_cor, tabular):
        """
        Args:
            img_ax: Axial Tri-Slab images (B, 3, 224, 224)
            img_cor: Coronal Tri-Slab images (B, 3, 224, 224)
            tabular: Clinical features (B, 8)

        Returns:
            Tensor of shape (B, 3) containing [alpha, sigma_base, sigma_growth]
        """
        # --- 1. Visual Branch Processing ---
        # Axial Branch
        # Backbone: (B, 3, 224, 224) -> (B, 1280, 7, 7)
        feat_ax = self.backbone_ax(img_ax)
        # GeM Pooling: (B, 1280, 7, 7) -> (B, 1280, 1, 1)
        feat_ax = self.gem_ax(feat_ax)
        # Flatten: (B, 1280)
        feat_ax = feat_ax.flatten(1)
        # Project: (B, D)
        token_ax = self.vis_proj(feat_ax)

        # Coronal Branch
        feat_cor = self.backbone_cor(img_cor)
        feat_cor = self.gem_cor(feat_cor).flatten(1)
        token_cor = self.vis_proj(feat_cor)

        # --- 2. Tabular Branch Processing ---
        # (B, 8) -> (B, D)
        token_tab = self.tab_mlp(tabular)

        # --- 3. Transformer Fusion ---
        # Stack tokens to form sequence: (B, 3, D)
        seq = torch.stack([token_ax, token_cor, token_tab], dim=1)

        # Apply Transformer Encoder
        out_seq = self.transformer(seq)

        # Extract the transformed tabular token (index 2)
        out_tab = out_seq[:, 2, :]

        # --- 4. Residual Anchor & Prediction ---
        # Concatenate transformed tabular token with the original raw tabular token
        # This preserves strong clinical priors (Baseline FVC, Percent)
        # Shape: (B, 2*D)
        combined = torch.cat([out_tab, token_tab], dim=1)

        # Predict parameters
        preds = self.head(combined)

        # --- 5. Output Formatting ---
        # Apply Softplus to sigmas to enforce positivity
        alpha = preds[:, 0]
        sigma_base = F.softplus(preds[:, 1])
        sigma_growth = F.softplus(preds[:, 2])

        # Return stacked parameters: (B, 3)
        return torch.stack([alpha, sigma_base, sigma_growth], dim=1)
