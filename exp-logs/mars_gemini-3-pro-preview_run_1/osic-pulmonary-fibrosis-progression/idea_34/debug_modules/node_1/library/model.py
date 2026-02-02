import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
import os
import numpy as np
from library.config import Config

# ==========================================
# 1. Components
# ==========================================


class VisualBackbone(nn.Module):
    """
    Independent High-Fidelity Visual Backbone.
    Uses EfficientNet-B0 initialized with ImageNet weights.
    Returns the native Global Average Pooled features (1280-dim).
    """

    def __init__(self):
        super().__init__()
        # num_classes=0 removes the classifier and returns the pooled features
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, num_classes=0
        )

    def forward(self, x):
        # x shape: (B, 3, 224, 224)
        # output shape: (B, 1280)
        return self.backbone(x)


class TabularGLU(nn.Module):
    """
    Gated Linear Unit for Tabular Expansion.
    Projects low-dim clinical features to high-dim visual space.
    Formula: V = (W1 x + b1) * sigmoid(W2 x + b2)
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        # x shape: (B, 4)
        h = self.fc(x)  # (B, output_dim * 2)
        a, b = h.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


# ==========================================
# 2. Main Model
# ==========================================


class ASADAN(nn.Module):
    """
    Anchored Symmetric-Attention Dual-Axis Network.
    Fuses Axial and Coronal views with Tabular data using Pre-Norm Attention.
    Predicts trajectory parameters (alpha, sigma_base, sigma_growth).
    """

    def __init__(self):
        super().__init__()

        # 1. Independent Backbones
        self.axial_net = VisualBackbone()
        self.coronal_net = VisualBackbone()

        # 2. Tabular Embedding
        # Inputs: Age, Sex, Smoking, Percent (4 features)
        self.tab_glu = TabularGLU(4, Config.FEATURE_DIM)

        # 3. Symmetric Attention Fusion
        # Pre-Norm Transformer Encoder Layer
        # We process a sequence of length 3: [Axial, Coronal, Tabular]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.FEATURE_DIM,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-Norm for stability
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4. Parametric Regression Head
        # Input: Fused Context (1280) + Raw Static Features (4)
        # We use a skip connection for raw features to anchor the priors
        self.head = nn.Sequential(
            nn.Linear(Config.FEATURE_DIM + 4, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3),  # Outputs: alpha, sigma_base, sigma_growth
        )

    def forward(self, img_axial, img_coronal, static_features, baseline_fvc, week):
        """
        Args:
            img_axial: (B, 3, 224, 224)
            img_coronal: (B, 3, 224, 224)
            static_features: (B, 4) [Age, Sex, Smoking, Percent]
            baseline_fvc: (B,) The FVC at week 0
            week: (B,) The relative week number to predict for
        Returns:
            fvc_pred: (B,)
            sigma_pred: (B,)
        """
        # 1. Extract Visual Features
        v_ax = self.axial_net(img_axial)  # (B, 1280)
        v_cor = self.coronal_net(img_coronal)  # (B, 1280)

        # 2. Expand Tabular Features
        v_tab = self.tab_glu(static_features)  # (B, 1280)

        # 3. Stack and Fuse
        # Sequence: [Axial, Coronal, Tabular] -> (B, 3, 1280)
        seq = torch.stack([v_ax, v_cor, v_tab], dim=1)

        # Apply Attention
        fused_seq = self.fusion(seq)  # (B, 3, 1280)

        # Global Average Pooling over the sequence dimension
        context = torch.mean(fused_seq, dim=1)  # (B, 1280)

        # 4. Predict Parameters
        # Concatenate context with raw static features for the head
        head_input = torch.cat([context, static_features], dim=1)

        params = self.head(head_input)  # (B, 3)

        alpha = params[:, 0]
        # Enforce positivity for uncertainty estimates
        sigma_base = F.softplus(params[:, 1]) + 1e-6
        sigma_growth = F.softplus(params[:, 2]) + 1e-6

        # 5. Trajectory Inference
        # FVC = Baseline + alpha * relative_week
        fvc_pred = baseline_fvc + alpha * week

        # Confidence = sigma_base + sigma_growth * |relative_week|
        sigma_pred = sigma_base + sigma_growth * torch.abs(week)

        return fvc_pred, sigma_pred


# ==========================================
# 3. Loss Function
# ==========================================


def laplace_log_likelihood_loss(
    y_true, y_pred, sigma, max_error=1000, confidence_clip=70
):
    """
    Differentiable loss function maximizing the modified Laplace Log Likelihood.
    Loss = -Metric
    Metric = - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
    Therefore: Loss = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)

    Includes clipping logic from the metric definition.
    """
    # Clip sigma to prevent singularity and adhere to metric definition
    sigma_clipped = torch.clamp(sigma, min=confidence_clip)

    # Calculate absolute error
    abs_error = torch.abs(y_true - y_pred)

    # Clip error at 1000ml to avoid outliers dominating gradients
    delta = torch.clamp(abs_error, max=max_error)

    sqrt_2 = math.sqrt(2)

    # Calculate negative log likelihood terms
    term1 = (sqrt_2 * delta) / sigma_clipped
    term2 = torch.log(sqrt_2 * sigma_clipped)

    loss = term1 + term2

    return torch.mean(loss)
