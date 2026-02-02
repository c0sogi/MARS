import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SLHDANetwork(nn.Module):
    """
    Shared-Latent Holistic Dual-Axis Network (SLH-DAN).

    Architecture:
    1. Two independent EfficientNet-B0 backbones for Axial and Coronal Tri-Slabs.
    2. Shared Latent Tabular Encoder (Deep MLP) for clinical metadata.
    3. Alignment projection to map latent tabular features to visual dimension.
    4. Pre-Norm Symmetric Attention (Transformer) for holistic fusion.
    5. Prior-Anchored Head predicting trajectory parameters (alpha, sigma_base, sigma_growth).
    """

    def __init__(self):
        super(SLHDANetwork, self).__init__()

        # ==========================================
        # 1. Independent Low-Capacity Visual Backbones
        # ==========================================
        # We use EfficientNet-B0 initialized with ImageNet weights.
        # num_classes=0 ensures we get the global pooled feature vector (1280-dim).
        self.backbone_ax = timm.create_model(
            Config.BACKBONE, pretrained=Config.BACKBONE_PRETRAINED, num_classes=0
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE, pretrained=Config.BACKBONE_PRETRAINED, num_classes=0
        )

        # Visual feature dimension (EfficientNet-B0 default)
        self.visual_dim = Config.VISUAL_DIM  # 1280

        # ==========================================
        # 2. Shared-Latent Tabular Encoder
        # ==========================================
        # Input features: [Norm_Age, Enc_Sex, Smoke_Ex, Smoke_Never, Smoke_Current, Norm_Percent]
        self.tabular_input_dim = 6
        self.latent_dim = Config.LATENT_DIM  # 128

        # Deep MLP: Linear -> GeLU -> Linear -> GeLU
        self.tabular_encoder = nn.Sequential(
            nn.Linear(self.tabular_input_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.latent_dim),
            nn.GELU(),
        )

        # ==========================================
        # 3. Bifurcated Flow & Alignment
        # ==========================================
        # Projects 128-dim latent vector to 1280-dim to match visual backbones for attention.
        self.alignment_layer = nn.Linear(self.latent_dim, self.visual_dim)

        # ==========================================
        # 4. Pre-Norm Symmetric Attention
        # ==========================================
        # Sequence: [Axial_Vector, Coronal_Vector, Aligned_Tabular_Vector]
        # We use a standard Transformer Encoder Layer.
        # norm_first=True enables Pre-Normalization for training stability.
        self.fusion_layer = nn.TransformerEncoderLayer(
            d_model=self.visual_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # ==========================================
        # 5. Balanced Prior-Anchored Head
        # ==========================================
        # Input: Concatenation of Holistic Fused Vector (1280) and Shared Latent Vector (128).
        # This skip connection preserves the clinical prior from being drowned out by visual noise.
        self.head_input_dim = self.visual_dim + self.latent_dim

        # Outputs 3 parameters: alpha (slope), sigma_base, sigma_growth
        self.head = nn.Linear(self.head_input_dim, 3)

    def forward(self, img_ax, img_cor, tab, weeks, base_fvc, base_week):
        """
        Forward pass implementing the SLH-DAN logic and Parametric Inference.

        Args:
            img_ax (torch.Tensor): Axial Tri-Slab images (B, 3, 224, 224).
            img_cor (torch.Tensor): Coronal Tri-Slab images (B, 3, 224, 224).
            tab (torch.Tensor): Tabular feature vectors (B, 6).
            weeks (torch.Tensor): The target week for prediction (B,).
            base_fvc (torch.Tensor): The baseline FVC measurement (B,).
            base_week (torch.Tensor): The week of the baseline measurement (B,).

        Returns:
            pred_fvc (torch.Tensor): Predicted FVC for the target week (B,).
            pred_sigma (torch.Tensor): Predicted Confidence for the target week (B,).
        """
        # --- 1. Feature Extraction ---
        # Visual Backbones -> (B, 1280)
        feat_ax = self.backbone_ax(img_ax)
        feat_cor = self.backbone_cor(img_cor)

        # Tabular Encoding -> (B, 128)
        t_lat = self.tabular_encoder(tab)

        # --- 2. Alignment & Tokenization ---
        # Project latent tabular to visual dimension -> (B, 1280)
        t_align = self.alignment_layer(t_lat)

        # Stack tokens to form sequence: [Axial, Coronal, Tabular]
        # Shape: (B, 3, 1280)
        tokens = torch.stack([feat_ax, feat_cor, t_align], dim=1)

        # --- 3. Holistic Fusion ---
        # Apply Self-Attention -> (B, 3, 1280)
        attended_tokens = self.fusion_layer(tokens)

        # Global Average Pooling across the sequence dimension (Holistic Readout)
        # Shape: (B, 1280)
        h_fused = torch.mean(attended_tokens, dim=1)

        # --- 4. Prediction Head ---
        # Concatenate Fused Context with original Latent Prior
        # Shape: (B, 1408)
        combined = torch.cat([h_fused, t_lat], dim=1)

        # Predict parameters -> (B, 3)
        params = self.head(combined)

        # Extract parameters
        # alpha: Slope of decline/incline (can be negative or positive)
        alpha = params[:, 0]

        # sigma_base: Uncertainty at t=0 (must be positive)
        sigma_base = F.softplus(params[:, 1])

        # sigma_growth: Uncertainty growth rate over time (must be positive)
        sigma_growth = F.softplus(params[:, 2])

        # --- 5. Parametric Inference (Anchored Trajectory) ---
        # Calculate time delta from baseline
        dt = weeks - base_week

        # Predict FVC: Linear trajectory anchored at baseline
        pred_fvc = base_fvc + alpha * dt

        # Predict Confidence: Linear growth of uncertainty based on time delta
        pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

        return pred_fvc, pred_sigma
