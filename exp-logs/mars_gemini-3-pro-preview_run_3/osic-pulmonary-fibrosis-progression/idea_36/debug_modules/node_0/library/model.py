import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ClinicalAnchor(nn.Module):
    """
    Stream A: Over-Parameterized Clinical Anchor.
    Learns the baseline disease trajectory from tabular data.
    """

    def __init__(self):
        super(ClinicalAnchor, self).__init__()
        # Architecture: Linear(Input -> 128) -> ReLU -> Linear(128 -> 64)
        self.feature_extractor = nn.Sequential(
            nn.Linear(Config.TABULAR_INPUT_DIM, Config.CLINICAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.CLINICAL_HIDDEN_DIM, Config.CLINICAL_LATENT_DIM),
        )

        # Base Predictions: Linear projection to mu and sigma_logit
        self.predictor = nn.Linear(Config.CLINICAL_LATENT_DIM, 2)

    def forward(self, x):
        # x shape: (Batch, TABULAR_INPUT_DIM)
        h_clin = self.feature_extractor(x)  # (Batch, CLINICAL_LATENT_DIM)
        preds = self.predictor(h_clin)  # (Batch, 2)

        mu_base = preds[:, 0]
        sigma_logit_base = preds[:, 1]

        return mu_base, sigma_logit_base, h_clin


class VisualResidual(nn.Module):
    """
    Stream B: Unregularized Cascaded Visual Residual.
    Learns a conditional residual correction based on images and clinical state.
    """

    def __init__(self):
        super(VisualResidual, self).__init__()

        # Backbone: EfficientNet-B2
        # num_classes=0 enables Global Average Pooling (GAP) automatically in timm
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            in_chans=3,
        )

        # --- Fine-Tuning Strategy ---
        # 1. Freeze everything initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze top two convolutional stages
        # In timm efficientnet, structure is: conv_stem -> bn1 -> blocks -> conv_head -> bn2 -> classifier
        # We unfreeze the head components and the last few blocks.

        # Unfreeze Head
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # Unfreeze last 2 blocks of the 'blocks' container
        num_blocks = len(self.backbone.blocks)
        for i in range(num_blocks - 2, num_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

        # Feature Dimension
        self.vis_dim = self.backbone.num_features

        # Projection to latent dim to balance fusion
        self.vis_projector = nn.Linear(self.vis_dim, Config.VISUAL_LATENT_DIM)

        # --- Cascaded Fusion & Projection ---
        # Input: Concatenation of Visual Latent and Clinical Latent
        fusion_dim = Config.VISUAL_LATENT_DIM + Config.CLINICAL_LATENT_DIM

        # Residual Head: Linear projection to delta_mu and delta_sigma
        self.residual_head = nn.Linear(fusion_dim, 2)

        # --- Structural Innovation 2: Zero Initialization ---
        # Initialize weights and biases to zero so the visual stream starts as a null-op
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

        # --- Structural Innovation 1: No Dropout ---
        # No dropout layers added.

    def forward(self, img, h_clin):
        # img shape: (Batch, 3, H, W)
        # h_clin shape: (Batch, CLINICAL_LATENT_DIM)

        # Extract Visual Features
        vis_feats = self.backbone(img)  # (Batch, vis_dim)

        # Project to latent space
        vis_latent = self.vis_projector(vis_feats)  # (Batch, VISUAL_LATENT_DIM)

        # Cascaded Fusion
        combined = torch.cat([vis_latent, h_clin], dim=1)

        # Predict Residuals
        residuals = self.residual_head(combined)

        delta_mu = residuals[:, 0]
        delta_sigma = residuals[:, 1]

        return delta_mu, delta_sigma


class UCOSRNet(nn.Module):
    """
    Unregularized Cascaded Output-Space Residual Network.
    Combines Clinical Anchor and Visual Residual streams.
    """

    def __init__(self):
        super(UCOSRNet, self).__init__()
        self.anchor = ClinicalAnchor()
        self.residual = VisualResidual()

    def forward(self, img, tabular):
        # --- Stream A: Clinical Anchor ---
        mu_base, sigma_logit_base, h_clin = self.anchor(tabular)

        # --- Stream B: Visual Residual ---
        delta_mu, delta_sigma = self.residual(img, h_clin)

        # --- Output Fusion ---
        # Mean: Additive correction
        mu_final = mu_base + delta_mu

        # Uncertainty: Additive correction in logit space, then Softplus
        # This ensures sigma is always positive and allows the residual to scale uncertainty
        sigma_logit_final = sigma_logit_base + delta_sigma
        sigma_final = F.softplus(sigma_logit_final) + 1e-6

        return mu_final, sigma_final
