import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SLHDAN(nn.Module):
    """
    Shared-Latent Holistic Dual-Axis Network (SLH-DAN).

    This model fuses two orthogonal CT views (Axial, Coronal) with clinical metadata
    to predict the trajectory of lung function decline.

    Architecture:
    1. Dual EfficientNet-B0 Backbones (Axial, Coronal) -> 1280-dim vectors.
    2. Shared Tabular Encoder -> 128-dim Latent Vector (Prior).
    3. Bifurcated Latent Flow:
       - Path A: Project to 1280-dim for Attention Fusion.
       - Path B: Skip connection to Head (Preserves Prior).
    4. Pre-Norm Symmetric Self-Attention for Multi-Modal Fusion.
    5. Parametric Head predicting (Slope, Sigma_Base, Sigma_Growth).
    """

    def __init__(self):
        super(SLHDAN, self).__init__()

        # ==========================================
        # 1. Visual Backbones (Independent)
        # ==========================================
        # EfficientNet-B0, ImageNet weights.
        # num_classes=0 ensures we get the pooled feature vector (1280-dim).
        self.backbone_ax = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0, global_pool="avg"
        )

        # Feature dimension for EfficientNet-B0
        self.visual_dim = Config.VISUAL_DIM  # 1280

        # ==========================================
        # 2. Shared Latent Tabular Encoder
        # ==========================================
        # Inputs: Age, Sex, Smoking, Percent (4 features)
        # Output: 128-dim Shared Latent Vector
        self.tabular_input_dim = 4
        self.latent_dim = Config.LATENT_DIM  # 128

        self.tab_encoder = nn.Sequential(
            nn.Linear(self.tabular_input_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.latent_dim),
            nn.GELU(),
        )

        # Projection for Fusion (128 -> 1280)
        self.tab_projector = nn.Linear(self.latent_dim, self.visual_dim)

        # ==========================================
        # 3. Contextual Fusion (Attention)
        # ==========================================
        # Pre-Norm Transformer Encoder Layer
        # Input sequence length: 3 (Axial, Coronal, Tabular)
        # Embedding dim: 1280
        self.fusion_layer = nn.TransformerEncoderLayer(
            d_model=self.visual_dim,
            nhead=4,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-Norm for stability
        )

        # ==========================================
        # 4. Parametric Head
        # ==========================================
        # Input: Fused Context (1280) + Shared Latent (128) = 1408
        self.head_input_dim = self.visual_dim + self.latent_dim

        self.head = nn.Sequential(
            nn.Linear(self.head_input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3),  # alpha, sigma_base, sigma_growth
        )

    def forward(
        self,
        img_ax,
        img_cor,
        tabular,
        base_fvc=None,
        base_week=None,
        current_week=None,
    ):
        """
        Args:
            img_ax: (B, 3, 224, 224) Axial Tri-Slab
            img_cor: (B, 3, 224, 224) Coronal Tri-Slab
            tabular: (B, 4) Clinical features
            base_fvc: (B, 1) [Optional] Baseline FVC for trajectory calc
            base_week: (B, 1) [Optional] Baseline Week
            current_week: (B, 1) [Optional] Target Week for prediction

        Returns:
            If temporal args are None:
                (B, 3) tensor containing [alpha, sigma_base, sigma_growth]
            If temporal args are provided:
                (B, 2) tensor containing [Predicted_FVC, Predicted_Sigma]
        """
        batch_size = img_ax.size(0)

        # ------------------------------------------
        # 1. Feature Extraction
        # ------------------------------------------
        # Visual Features (B, 1280)
        feat_ax = self.backbone_ax(img_ax)
        feat_cor = self.backbone_cor(img_cor)

        # Tabular Latent (B, 128)
        # This is the "Clinical Prior"
        lat_tab = self.tab_encoder(tabular)

        # ------------------------------------------
        # 2. Bifurcation & Alignment
        # ------------------------------------------
        # Flow A: Align for Fusion (B, 1280)
        feat_tab_aligned = self.tab_projector(lat_tab)

        # Stack tokens: [Axial, Coronal, Tabular] -> (B, 3, 1280)
        tokens = torch.stack([feat_ax, feat_cor, feat_tab_aligned], dim=1)

        # ------------------------------------------
        # 3. Contextual Fusion
        # ------------------------------------------
        # Apply Self-Attention
        # Output: (B, 3, 1280)
        contextualized_tokens = self.fusion_layer(tokens)

        # Holistic Readout: GAP across sequence dimension
        # (B, 1280)
        h_fused = torch.mean(contextualized_tokens, dim=1)

        # ------------------------------------------
        # 4. Prediction Head
        # ------------------------------------------
        # Concatenate Fused Context with Original Latent (Flow B)
        # (B, 1280 + 128) -> (B, 1408)
        combined = torch.cat([h_fused, lat_tab], dim=1)

        # Predict Parameters
        # raw_out: [alpha, raw_sigma_base, raw_sigma_growth]
        raw_out = self.head(combined)

        # Apply Activations
        # alpha (Slope) is unbounded (can be negative for decline)
        # sigma (Confidence) must be positive -> Softplus
        alpha = raw_out[:, 0].view(-1, 1)
        sigma_base = F.softplus(raw_out[:, 1]).view(-1, 1)
        sigma_growth = F.softplus(raw_out[:, 2]).view(-1, 1)

        # ------------------------------------------
        # 5. Trajectory Inference (Optional)
        # ------------------------------------------
        # If we have the temporal anchors, compute the actual FVC and Confidence
        if base_fvc is not None and base_week is not None and current_week is not None:
            # Calculate delta time
            dt = current_week - base_week

            # Linear Trajectory: FVC = Base + alpha * dt
            pred_fvc = base_fvc + (alpha * dt)

            # Confidence Trajectory: Sigma = Base + Growth * |dt|
            pred_sigma = sigma_base + (sigma_growth * torch.abs(dt))

            # Return (FVC, Confidence)
            return torch.cat([pred_fvc, pred_sigma], dim=1)

        # Otherwise return the static parameters
        return torch.cat([alpha, sigma_base, sigma_growth], dim=1)
