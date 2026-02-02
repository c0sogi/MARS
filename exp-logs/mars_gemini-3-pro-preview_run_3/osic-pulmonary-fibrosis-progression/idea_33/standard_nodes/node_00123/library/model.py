import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPRNet).
    Cite {solution_lesson_node_00052}: Implements a Deep Interaction Stream and a Linear Residual Stream.
    Cite {solution_lesson_node_00060}: Fuses streams in a shared latent space (Latent Fusion).
    Cite {solution_lesson_node_00118}: Uses standard initialization (no zero-init).
    """

    def __init__(self):
        super(DSPRNet, self).__init__()

        # 1. Backbone: EfficientNet-B2
        weights = models.EfficientNet_B2_Weights.DEFAULT if Config.PRETRAINED else None
        self.backbone = models.efficientnet_b2(weights=weights)
        self.backbone.classifier = nn.Identity()

        # Freezing Strategy (Cite {solution_lesson_node_00027}: Differential Learning Rates support)
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Unfreeze top two blocks
        for i in range(7, 9):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True

        self.feature_dim = 1408
        self.latent_dim = 64

        # 2. Deep Interaction Stream (Stream A)
        # Image + Tabular -> MLP -> Latent
        self.deep_projector = nn.Sequential(
            nn.Linear(self.feature_dim + Config.CLINICAL_INPUT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, self.latent_dim),
        )

        # 3. Linear Residual Stream (Stream B)
        # Tabular -> Linear -> Latent
        # Cite {solution_lesson_node_00060}: Over-parameterize linear stream to latent dim
        self.linear_projector = nn.Linear(Config.CLINICAL_INPUT_DIM, self.latent_dim)

        # 4. Shared Head
        # Latent -> Output
        self.head = nn.Linear(self.latent_dim, 2)

    def forward(self, images, tabular_input):
        # Stream A: Deep Interaction
        x = self.backbone.features(images)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        # Cite {solution_lesson_node_00103}: Early Fusion
        x_fused = torch.cat([x, tabular_input], dim=1)
        latent_deep = self.deep_projector(x_fused)

        # Stream B: Linear Residual
        latent_linear = self.linear_projector(tabular_input)

        # Latent Fusion (Summation)
        latent_final = latent_deep + latent_linear

        # Output Projection
        out = self.head(latent_final)

        mu = out[:, 0]
        sigma_raw = out[:, 1]
        sigma = F.softplus(sigma_raw) + 1e-6

        return mu, sigma
