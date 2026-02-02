import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPRNet).
    Stream A: Deep Interaction (Image + Tabular) -> Non-Linear
    Stream B: Linear Projection (Baseline FVC + Time) -> Linear
    Fused in latent space before final projection.
    """

    def __init__(self):
        super(DSPRNet, self).__init__()

        # 1. Backbone: EfficientNet-B2
        weights = models.EfficientNet_B2_Weights.DEFAULT if Config.PRETRAINED else None
        self.backbone = models.efficientnet_b2(weights=weights)
        self.backbone.classifier = nn.Identity()

        # Feature Extraction Setup
        self.feature_dim = 1408

        # Freezing Strategy
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze top two blocks (7 and 8)
        for i in range(7, 9):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True

        # 2. Stream A: Deep Interaction Stream
        # Input: Image Features (1408) + All Tabular Features (5)
        self.deep_stream = nn.Sequential(
            nn.Linear(self.feature_dim + Config.CLINICAL_INPUT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
        )

        # 3. Stream B: Linear Projection Stream
        # Input: Baseline FVC (1) + Relative Time (1) = 2
        # Projects to same latent dim (64) to avoid bottlenecking
        self.linear_stream = nn.Linear(2, 64)

        # 4. Shared Head
        self.head = nn.Linear(64, 2)  # [mu, sigma_logit]

    def forward(self, images, tabular_input):
        # Extract visual features
        x = self.backbone.features(images)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)

        # Stream A: Deep Interaction
        deep_in = torch.cat([x, tabular_input], dim=1)
        deep_out = self.deep_stream(deep_in)

        # Stream B: Linear Projection
        # tabular_input: [BaseFVC, Time, Age, Sex, Smoke]
        # We take BaseFVC (idx 0) and Time (idx 1)
        linear_in = tabular_input[:, :2]
        linear_out = self.linear_stream(linear_in)

        # Latent Fusion (Summation)
        fused = deep_out + linear_out

        # Final Projection
        out = self.head(fused)

        mu = out[:, 0]
        sigma = F.softplus(out[:, 1]) + 1e-6

        return mu, sigma
