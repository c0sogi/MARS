import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPRNet).
    Cite solution_lesson_node_00052: Dual-Stream Residuals for Strong Autoregressive Signals.
    Cite solution_lesson_node_00060: Over-Parameterization of Linear Baselines.
    """

    def __init__(self):
        super(DSPRNet, self).__init__()

        # Dimensions
        self.latent_dim = 64
        self.dropout_rate = Config.DROPOUT_RATE
        self.img_feature_dim = 1408  # EfficientNet-B2
        self.tabular_dim = 5  # [Base_FVC, Rel_Time, Age, Sex, Smoke]

        # Stream A: Linear Residual (Baseline + Time)
        # Projects dominant autoregressive features to latent space
        # Cite solution_lesson_node_00060: Over-parameterize linear stream in latent space
        self.linear_stream = nn.Linear(2, self.latent_dim)

        # Stream B: Deep Interaction (Image + Tabular)
        # Cite solution_lesson_node_00035: Direct concatenation of strong scalar predictors
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # Freeze/Unfreeze strategy
        # Cite solution_lesson_node_00027: Differential Learning Rates / Fine-tuning
        for param in self.backbone.parameters():
            param.requires_grad = False
        for name, param in self.backbone.named_parameters():
            if any(x in name for x in ["blocks.5", "blocks.6", "conv_head", "bn2"]):
                param.requires_grad = True

        self.deep_head = nn.Sequential(
            nn.Linear(self.img_feature_dim + self.tabular_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, self.latent_dim),
        )

        # Shared Head
        # Cite solution_lesson_node_00055: Preserve Low-Complexity Pathways for Uncertainty
        self.final_head = nn.Linear(self.latent_dim, 2)

    def forward(self, tabular, image):
        # Stream A: Linear (Baseline FVC, Relative Time)
        # tabular is [Base, Time, Age, Sex, Smoke]. We take first 2.
        linear_latent = self.linear_stream(tabular[:, :2])

        # Stream B: Deep
        img_feats = self.backbone(image)
        # Concatenate all tabular features with image features
        combined = torch.cat([img_feats, tabular], dim=1)
        deep_latent = self.deep_head(combined)

        # Latent Summation
        # Cite solution_lesson_node_00052: Sum in latent space
        fused_latent = linear_latent + deep_latent

        # Prediction
        preds = self.final_head(fused_latent)
        mu = preds[:, 0]
        sigma = F.softplus(preds[:, 1]) + 1e-6

        return mu, sigma
