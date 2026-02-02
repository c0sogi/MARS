import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class TabularMLP(nn.Module):
    """
    Deep MLP to project raw tabular features into the visual latent space.
    Structure: Linear -> GeLU -> Linear -> GeLU -> Linear
    """

    def __init__(self, input_dim=4, output_dim=1280, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SCVRNet(nn.Module):
    """
    Scaled Contextualized Visual-Residual Network (SCVR-Net).

    Architecture:
    1. Dual Independent Visual Backbones (EfficientNet-B1) for Axial and Coronal views.
    2. Deep Tabular MLP for metadata embedding.
    3. Pre-Norm Symmetric Attention for cross-modal contextualization.
    4. Visual-Exclusive Pooling (isolating visual delta).
    5. Prior-Anchored Parametric Head predicting alpha, sigma_base, sigma_growth.
    """

    def __init__(self):
        super().__init__()

        # 1. Independent Scaled Visual Backbones
        # EfficientNet-B1 outputs 1280 dim features with num_classes=0
        self.backbone_ax = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        self.backbone_cor = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # Feature dimension for EfficientNet-B1
        self.feature_dim = Config.BACKBONE_DIM

        # 2. Robust Deep Tabular Expansion
        # Input: Age, Sex, Smoking, Percent (4 features)
        self.tabular_mlp = TabularMLP(
            input_dim=4,
            output_dim=self.feature_dim,
            hidden_dim=Config.TABULAR_HIDDEN_DIM // 2,  # Intermediate dim
        )

        # 3. Pre-Norm Symmetric Attention (Contextualization)
        # We use a Transformer Encoder Layer with norm_first=True for Pre-Norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=Config.NUM_ATTENTION_HEADS,
            dim_feedforward=self.feature_dim * 2,
            dropout=Config.DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 4. Prior-Anchored Parametric Head
        # Input: Contextualized Visual Residual (1280) + Raw Tabular (4)
        head_input_dim = self.feature_dim + 4

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3),  # alpha, sigma_base, sigma_growth
        )

    def forward(self, img_ax, img_cor, tabular):
        """
        Args:
            img_ax: (B, 3, 240, 240)
            img_cor: (B, 3, 240, 240)
            tabular: (B, 4) - [Age, Sex, Smoking, Percent]
        Returns:
            preds: (B, 3) - [alpha, sigma_base, sigma_growth]
        """
        batch_size = img_ax.size(0)

        # --- 1. Feature Extraction ---
        # Get visual tokens (B, 1280)
        v_ax = self.backbone_ax(img_ax)
        v_cor = self.backbone_cor(img_cor)

        # Get tabular token (B, 1280)
        v_tab = self.tabular_mlp(tabular)

        # --- 2. Contextualization ---
        # Stack tokens: [V_ax, V_cor, V_tab] -> (B, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, v_tab], dim=1)

        # Apply Self-Attention
        contextualized_tokens = self.context_attention(tokens)

        # --- 3. Visual-Exclusive Pooled Readout ---
        # Extract updated visual tokens only (indices 0 and 1)
        # Discard the tabular token (index 2) for the residual calculation
        v_ax_prime = contextualized_tokens[:, 0, :]
        v_cor_prime = contextualized_tokens[:, 1, :]

        # Average Pooling of visual tokens
        visual_residual = (v_ax_prime + v_cor_prime) / 2.0

        # --- 4. Prediction ---
        # Concatenate Visual Residual with Raw Tabular Features (Skip Connection)
        combined = torch.cat([visual_residual, tabular], dim=1)

        # Predict parameters
        out = self.head(combined)

        # Extract components
        alpha = out[:, 0]  # Slope (unbounded)
        sigma_base = F.softplus(out[:, 1])  # Positive confidence intercept
        sigma_growth = F.softplus(out[:, 2])  # Positive confidence slope

        return torch.stack([alpha, sigma_base, sigma_growth], dim=1)
