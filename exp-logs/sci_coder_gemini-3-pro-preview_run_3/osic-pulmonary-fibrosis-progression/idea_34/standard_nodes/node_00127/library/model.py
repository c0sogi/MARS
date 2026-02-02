import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ClinicalAnchor(nn.Module):
    """
    Stream A: Over-Parameterized Clinical Anchor.
    Acts as the base learner using tabular data.
    """

    def __init__(self, input_dim=5, hidden_dim=128, latent_dim=64):
        super(ClinicalAnchor, self).__init__()

        # MLP Backbone
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            # Note: No ReLU here on the latent output to allow full vector space usage
            # before fusion or projection.
        )

        # Base Prediction Head (Mu, Raw Sigma)
        self.head = nn.Linear(latent_dim, 2)

    def forward(self, x):
        # x shape: (Batch, 5)
        latent = self.net(x)  # (Batch, 64)
        base_preds = self.head(latent)  # (Batch, 2) -> [mu_base, sigma_logit_base]
        return latent, base_preds


class VisualResidual(nn.Module):
    """
    Stream B: Cascaded Visual Residual.
    Extracts visual features and learns a residual correction conditioned on the clinical state.
    """

    def __init__(self, clinical_latent_dim=64, dropout_rate=0.2):
        super(VisualResidual, self).__init__()

        # 1. Backbone: EfficientNet-B2
        # num_classes=0 removes the classifier, returning pooled features if global_pool is set
        # However, timm returns features before pooling if global_pool='' or num_classes=0 usually.
        # We use forward_features + manual pooling or let timm handle it.
        # EfficientNet B2 feature dim is 1408.
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # Feature dimension for EfficientNet-B2
        self.img_feature_dim = self.backbone.num_features  # 1408

        # 2. Freezing Strategy (Unfreeze top 2 stages)
        # EfficientNet blocks are named 'blocks.0' through 'blocks.6'.
        # We unfreeze blocks 5, 6 and the conv_head.
        for param in self.backbone.parameters():
            param.requires_grad = False

        for name, param in self.backbone.named_parameters():
            # Unfreeze top stages (blocks.5, blocks.6) and head components
            if any(x in name for x in ["blocks.5", "blocks.6", "conv_head", "bn2"]):
                param.requires_grad = True

        # 3. Fusion & Residual Head
        # Concatenate Image Features + Clinical Latent
        fusion_dim = self.img_feature_dim + clinical_latent_dim

        self.residual_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            # Cite Lesson 126: Removed Dropout to preserve weak residual signals
            nn.Linear(64, 2),  # [delta_mu, delta_sigma_logit]
        )

        # Cite Lesson 118: Removed Zero Initialization to prevent bias towards random linear projection

    def forward(self, img, clinical_latent):
        # img: (Batch, 3, 260, 260)
        # clinical_latent: (Batch, 64)

        # Extract visual features
        # timm with num_classes=0 and global_pool='avg' returns (Batch, num_features)
        img_feats = self.backbone(img)

        # Fusion
        fused = torch.cat([img_feats, clinical_latent], dim=1)

        # Predict Residuals
        residuals = self.residual_head(fused)  # (Batch, 2)

        return residuals


class MACOSR(nn.Module):
    """
    Metric-Aligned Cascaded Output-Space Residual Network.
    Combines Clinical Anchor and Visual Residual streams.
    """

    def __init__(self):
        super(MACOSR, self).__init__()

        # Input dim is 5 based on library.data processing:
        # [Base_FVC, Rel_Time, Age, Sex, Smoke]
        self.clinical_input_dim = 5

        self.clinical_anchor = ClinicalAnchor(
            input_dim=self.clinical_input_dim,
            hidden_dim=Config.CLINICAL_HIDDEN_DIM,
            latent_dim=Config.CLINICAL_LATENT_DIM,
        )

        self.visual_residual = VisualResidual(
            clinical_latent_dim=Config.CLINICAL_LATENT_DIM,
            dropout_rate=Config.DROPOUT_RATE,
        )

    def forward(self, tabular, image):
        """
        Args:
            tabular: (Batch, 5) tensor of clinical features.
            image: (Batch, 3, H, W) tensor of slice volumes.
        Returns:
            mu: Predicted FVC (normalized scale).
            sigma: Predicted Confidence (normalized scale).
        """
        # Stream A: Clinical Anchor
        clinical_latent, base_preds = self.clinical_anchor(tabular)
        # base_preds: [mu_base, sigma_logit_base]

        # Stream B: Visual Residual
        # Pass latent to visual stream for conditioned correction
        residuals = self.visual_residual(image, clinical_latent)
        # residuals: [delta_mu, delta_sigma_logit]

        # Output Space Summation
        mu_final = base_preds[:, 0] + residuals[:, 0]
        sigma_logit_final = base_preds[:, 1] + residuals[:, 1]

        # Uncertainty Activation
        # Softplus ensures positivity. Epsilon prevents division by zero in loss.
        # We do not clip to 70 here (that is for metric/submission), allowing gradient flow.
        sigma_final = F.softplus(sigma_logit_final) + 1e-6

        return mu_final, sigma_final
