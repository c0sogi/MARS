import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class CAPNet(nn.Module):
    """
    Content-Adaptive Parametric Trajectory Network (CAP-Net).

    Predicts the parameters of a linear trajectory (alpha, beta, gamma) and uncertainty (delta)
    based on a baseline CT scan (3 slices) and clinical metadata.

    The network predicts coefficients that are applied to the temporal variable 'Weeks'
    and the 'Baseline FVC' to dynamically compute the predicted FVC.
    """

    def __init__(self):
        super().__init__()

        # --- 1. Image Branch (Content-Adaptive 2.5D) ---
        # Load frozen EfficientNet-B0 backbone
        # num_classes=0 returns the global pool features (flat vector)
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=True, num_classes=0
        )

        # Freeze backbone weights to preserve semantic features
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Get input features dimension (1280 for EfficientNet-B0)
        in_features = self.backbone.num_features

        # Project image features to lower dimension to balance modalities
        self.img_projector = nn.Linear(in_features, Config.PROJECTION_DIM)

        # --- 2. Metadata Embeddings ---
        # Sex: 2 classes (Male, Female) -> Embedding dim 4
        self.sex_embed = nn.Embedding(2, 4)
        # SmokingStatus: 3 classes -> Embedding dim 4
        self.smoke_embed = nn.Embedding(3, 4)

        # --- 3. Fusion & Parameter Head ---
        # Calculate fusion dimension:
        # Image Projection (128) + Sex (4) + Smoking (4) + Age (1) + Percent (1) + Baseline FVC (1)
        # Note: Age and Percent are combined in meta_num (dim 2)
        fusion_dim = Config.PROJECTION_DIM + 4 + 4 + 2 + 1

        # MLP to predict trajectory parameters
        # No Batch Normalization and No Dropout to preserve linear signal precision
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, 4),  # Outputs: alpha, beta, gamma, delta
        )

    def forward(self, image, meta_cat, meta_num, baseline_fvc_scaled, weeks_scaled):
        """
        Args:
            image: (B, 3, H, W) - 3 slices treated as RGB channels
            meta_cat: (B, 2) - [Sex_idx, Smoking_idx]
            meta_num: (B, 2) - [Age_scaled, Percent_scaled]
            baseline_fvc_scaled: (B,) - Standardized baseline FVC
            weeks_scaled: (B,) - Standardized weeks

        Returns:
            mu: (B,) - Predicted FVC (Standardized)
            sigma: (B,) - Predicted Confidence (Standardized)
        """
        # --- Image Feature Extraction ---
        # (B, 3, H, W) -> (B, 1280)
        img_feats = self.backbone(image)
        # Project: (B, 1280) -> (B, 128)
        img_feats = self.img_projector(img_feats)

        # --- Metadata Embedding ---
        # meta_cat[:, 0] is Sex, meta_cat[:, 1] is Smoking
        sex_emb = self.sex_embed(meta_cat[:, 0])  # (B, 4)
        smoke_emb = self.smoke_embed(meta_cat[:, 1])  # (B, 4)

        # --- Fusion ---
        # Ensure baseline_fvc_scaled is (B, 1) for concatenation
        if baseline_fvc_scaled.dim() == 1:
            baseline_fvc_scaled_in = baseline_fvc_scaled.unsqueeze(1)
        else:
            baseline_fvc_scaled_in = baseline_fvc_scaled

        # Concatenate all features
        fusion_vec = torch.cat(
            [img_feats, sex_emb, smoke_emb, meta_num, baseline_fvc_scaled_in], dim=1
        )

        # --- Parameter Prediction ---
        params = self.mlp(fusion_vec)  # (B, 4)

        alpha = params[:, 0]
        beta = params[:, 1]
        gamma = params[:, 2]
        delta = params[:, 3]

        # --- Trajectory Logic ---
        # Equation: mu = alpha * Baseline + beta + gamma * Weeks
        # We use the standardized inputs here.
        # Ensure dimensionality matches for broadcasting

        mu = (alpha * baseline_fvc_scaled) + beta + (gamma * weeks_scaled)

        # --- Uncertainty ---
        # sigma = softplus(delta) + epsilon
        # We do not clip to 70 here; clipping is done in metric/post-processing
        sigma = F.softplus(delta) + 1e-3

        return mu, sigma
