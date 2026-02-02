import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class TabularMLP(nn.Module):
    """
    Deep Multi-Layer Perceptron to project raw clinical features into a high-dimensional manifold.
    Structure: Linear -> GeLU -> Linear -> GeLU -> Linear
    """

    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(TabularMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class HiFiDACR(nn.Module):
    """
    High-Fidelity Deep-Aligned Contextualized-Residual Network (HiFi-DACR).

    Architecture:
    1. Dual EfficientNet-B1 Backbones (Axial & Coronal) -> 1280 dim.
    2. Deep Tabular Alignment (MLP) -> 1280 dim.
    3. Pre-Norm Symmetric Attention for multi-modal fusion.
    4. Visual-Exclusive Pooled Readout (isolating visual delta).
    5. Prior-Anchored Parametric Head (predicts trajectory params).
    """

    def __init__(self, tab_input_dim=7):
        """
        Args:
            tab_input_dim (int): Dimension of the raw tabular features.
                                 Default 7 (2 numeric + 2 Sex + 3 Smoking).
        """
        super(HiFiDACR, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Independent Scaled Visual Backbones
        # ---------------------------------------------------------------------
        # We use EfficientNet-B1 which outputs 1280 features at the final stage.
        # We load pretrained weights for better convergence.
        self.backbone_ax = models.efficientnet_b1(weights="DEFAULT")
        self.backbone_cor = models.efficientnet_b1(weights="DEFAULT")

        # We will use the features and avgpool layers directly, ignoring the classifier.
        self.vis_dim = Config.VISUAL_DIM  # 1280

        # ---------------------------------------------------------------------
        # 2. Deep Tabular Alignment
        # ---------------------------------------------------------------------
        self.tab_mlp = TabularMLP(
            input_dim=tab_input_dim,
            output_dim=Config.TABULAR_DIM,  # 1280
            hidden_dim=512,
        )

        # ---------------------------------------------------------------------
        # 3. Pre-Norm Symmetric Attention
        # ---------------------------------------------------------------------
        # LayerNorm before Attention
        self.norm1 = nn.LayerNorm(Config.VISUAL_DIM)
        self.attn = nn.MultiheadAttention(
            embed_dim=Config.VISUAL_DIM,
            num_heads=Config.NUM_ATTENTION_HEADS,
            dropout=Config.DROPOUT,
            batch_first=True,
        )

        # LayerNorm before FFN
        self.norm2 = nn.LayerNorm(Config.VISUAL_DIM)
        self.ffn = nn.Sequential(
            nn.Linear(Config.VISUAL_DIM, Config.FFN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.FFN_DIM, Config.VISUAL_DIM),
            nn.Dropout(Config.DROPOUT),
        )

        # ---------------------------------------------------------------------
        # 4. Prior-Anchored Parametric Head
        # ---------------------------------------------------------------------
        # Input: Contextualized Visual Residual (1280) + Raw Tabular (tab_input_dim)
        # We use a skip connection for raw tabular data to preserve priors.
        head_input_dim = Config.VISUAL_DIM + tab_input_dim

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3),  # Outputs: alpha, sigma_base, sigma_growth
        )

    def forward_backbone(self, backbone, x):
        """Helper to run efficientnet features + avgpool + flatten"""
        x = backbone.features(x)
        x = backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(
        self, image_axial, image_coronal, tabular, baseline_fvc, relative_week, **kwargs
    ):
        """
        Args:
            image_axial (Tensor): (B, 3, 240, 240)
            image_coronal (Tensor): (B, 3, 240, 240)
            tabular (Tensor): (B, tab_input_dim) - Raw scaled/encoded features
            baseline_fvc (Tensor): (B,) or (B, 1)
            relative_week (Tensor): (B,) or (B, 1)

        Returns:
            outputs (Tensor): (B, 2) -> [FVC_pred, Confidence_pred]
        """
        # --- 1. Feature Extraction ---
        # Visual Features
        v_ax = self.forward_backbone(self.backbone_ax, image_axial)  # (B, 1280)
        v_cor = self.forward_backbone(self.backbone_cor, image_coronal)  # (B, 1280)

        # Tabular Embedding
        v_tab = self.tab_mlp(tabular)  # (B, 1280)

        # --- 2. Contextualization (Attention) ---
        # Stack tokens: [Axial, Coronal, Tabular] -> (B, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, v_tab], dim=1)

        # Pre-Norm Attention Block
        # 1. Norm -> Attn -> Residual Add
        tokens_norm = self.norm1(tokens)
        attn_out, _ = self.attn(tokens_norm, tokens_norm, tokens_norm)
        tokens = tokens + attn_out

        # 2. Norm -> FFN -> Residual Add
        tokens_norm2 = self.norm2(tokens)
        ffn_out = self.ffn(tokens_norm2)
        tokens = tokens + ffn_out

        # --- 3. Visual-Exclusive Pooled Readout ---
        # We discard the tabular token (index 2) for the pooling operation
        # to isolate the "visual delta" information.
        v_ax_prime = tokens[:, 0, :]
        v_cor_prime = tokens[:, 1, :]

        # Average Pool the visual tokens
        v_res = (v_ax_prime + v_cor_prime) / 2.0  # (B, 1280)

        # --- 4. Prior-Anchored Parametric Head ---
        # Concatenate with RAW tabular features (Skip Connection)
        head_input = torch.cat([v_res, tabular], dim=1)

        # Predict parameters
        params = self.head(head_input)  # (B, 3)

        alpha = params[:, 0]
        sigma_base_raw = params[:, 1]
        sigma_growth_raw = params[:, 2]

        # Apply constraints
        # Sigma must be positive -> Softplus
        sigma_base = F.softplus(sigma_base_raw)
        sigma_growth = F.softplus(sigma_growth_raw)

        # --- 5. Inference Logic ---
        # Ensure correct shapes for broadcasting
        if baseline_fvc.dim() == 1:
            baseline_fvc = baseline_fvc.view(-1)
        if relative_week.dim() == 1:
            relative_week = relative_week.view(-1)

        # Calculate FVC: Baseline + slope * time
        fvc_pred = baseline_fvc + alpha * relative_week

        # Calculate Confidence: Base_uncertainty + growth * |time|
        confidence_pred = sigma_base + sigma_growth * torch.abs(relative_week)

        # Stack for output (B, 2)
        outputs = torch.stack([fvc_pred, confidence_pred], dim=1)

        return outputs
