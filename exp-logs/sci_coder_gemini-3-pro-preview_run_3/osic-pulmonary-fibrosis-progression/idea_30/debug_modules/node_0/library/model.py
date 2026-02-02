import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class MAZR_DS(nn.Module):
    """
    Metric-Aligned Zero-Residual Dual-Stream Network (MAZR-DS).

    This model implements a hybrid architecture where a visual stream acts as a
    perturbative residual to a robust clinical anchor stream.

    Structure:
    1. Backbone: EfficientNet-B2 (Pretrained, top stages unfrozen).
    2. Stream A: Clinical Anchor MLP (Baseline FVC + Meta -> Latent).
    3. Stream B: Visual Interaction MLP (Image + Meta -> Zero-Init Residual).
    4. Fusion: Summation of Stream A and B.
    5. Head: Shared projection to Mu and Sigma.
    """

    def __init__(self):
        super(MAZR_DS, self).__init__()

        # =====================================================================
        # 1. Image Backbone (EfficientNet-B2)
        # =====================================================================
        # Load pretrained model, remove classifier (num_classes=0)
        # This returns the global pooled features (B, num_features)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0
        )

        # Feature dimension for B2 is typically 1408
        self.n_features = self.backbone.num_features

        # Freeze all parameters first to preserve low-level feature detectors
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze top two convolutional stages (blocks.5, blocks.6) and the conv_head
        # This allows domain adaptation for high-level features (e.g., fibrosis patterns)
        for name, param in self.backbone.named_parameters():
            if any(x in name for x in ["blocks.5", "blocks.6", "conv_head", "bn2"]):
                param.requires_grad = True

        # Projection layer to map high-dim image features to shared latent space
        self.img_projector = nn.Linear(self.n_features, Config.IMG_PROJ_DIM)

        # =====================================================================
        # 2. Stream A: Clinical Anchor (Over-Parameterized MLP)
        # =====================================================================
        # Input: [Baseline_FVC, t_rel, Age, Sex, Smoking] -> 5 features
        # This stream learns the "Expected Clinical Trajectory"
        self.stream_a = nn.Sequential(
            nn.Linear(5, Config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.MLP_HIDDEN_DIM, Config.IMG_PROJ_DIM),
        )

        # =====================================================================
        # 3. Stream B: Visual Interaction (Zero-Residual)
        # =====================================================================
        # Input: Image Projection (64) + Tabular (5) -> 69 features
        # This stream learns the deviation from the expected trajectory based on the scan
        self.stream_b = nn.Sequential(
            nn.Linear(Config.IMG_PROJ_DIM + 5, Config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.MLP_HIDDEN_DIM, Config.IMG_PROJ_DIM),
        )

        # Zero Initialization
        # We initialize the weights and bias of the final linear layer of Stream B to zero.
        # This ensures that at epoch 0, Stream B outputs 0, and the model behaves exactly
        # like the Clinical Anchor (Stream A). This prevents high-dimensional image noise
        # from destabilizing the training in the early phases.
        nn.init.zeros_(self.stream_b[-1].weight)
        nn.init.zeros_(self.stream_b[-1].bias)

        # =====================================================================
        # 4. Shared Head
        # =====================================================================
        # Projects the fused latent vector to Mu and Raw Sigma
        self.head = nn.Linear(Config.IMG_PROJ_DIM, 2)

    def forward(self, imgs, tabular):
        """
        Forward pass of the MAZR-DS network.

        Args:
            imgs: Tensor of shape (B, 3, H, W). The 3 channels represent 3 selected slices.
            tabular: Tensor of shape (B, 5). [BaseFVC_norm, t_rel, Age_norm, Sex, Smoke]

        Returns:
            mu: Predicted FVC (normalized scale)
            sigma: Predicted Confidence (normalized scale, positive)
        """
        # --- Image Branch ---
        # Extract features: (B, 1408)
        img_feats = self.backbone(imgs)
        # Project: (B, 64)
        img_emb = self.img_projector(img_feats)

        # --- Stream A: Clinical Anchor ---
        # Learns the base trajectory from clinical data
        out_a = self.stream_a(tabular)  # (B, 64)

        # --- Stream B: Visual Interaction ---
        # Learns the residual correction based on image content
        # Early Fusion: Concat image embedding and tabular data
        fusion_input = torch.cat([img_emb, tabular], dim=1)  # (B, 69)
        out_b = self.stream_b(fusion_input)  # (B, 64)

        # --- Latent Fusion ---
        # Summation: Anchor + Residual
        # At init, out_b is 0, so h_final = out_a
        h_final = out_a + out_b  # (B, 64)

        # --- Prediction Head ---
        logits = self.head(h_final)  # (B, 2)

        mu = logits[:, 0]
        raw_sigma = logits[:, 1]

        # Enforce positivity for sigma using softplus
        # Add epsilon to prevent numerical instability and ensure non-zero sigma
        sigma = F.softplus(raw_sigma) + 1e-6

        return mu, sigma
