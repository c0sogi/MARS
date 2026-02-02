import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPRNet).
    Cite solution_lesson_node_00052: Dual-Stream architecture with Latent Fusion.
    Cite solution_lesson_node_00060: Over-parameterized linear stream.
    """

    def __init__(self):
        super().__init__()

        # 1. Visual Backbone (EfficientNet-B2)
        # Cite solution_lesson_node_00071: Use B2 for capacity and resolution
        weights = models.EfficientNet_B2_Weights.DEFAULT
        self.backbone = models.efficientnet_b2(weights=weights)
        self.feature_dim = self.backbone.classifier[1].in_features

        # Freezing Logic (Cite solution_lesson_node_00027)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.features[-1].parameters():
            param.requires_grad = True
        for param in self.backbone.features[-2].parameters():
            param.requires_grad = True

        # Dimensions
        n_tabular = 5  # Base_FVC, Time, Age, Sex, Smoke
        n_linear = 2  # Base_FVC, Time (Cite solution_lesson_node_00052)
        latent_dim = 64

        # 2. Deep Interaction Stream (Early Fusion)
        # Cite solution_lesson_node_00103: Early Fusion of Tabular + Image
        # Cite solution_lesson_node_00139: Critical context visible to deep branch
        self.deep_mlp = nn.Sequential(
            nn.Linear(self.feature_dim + n_tabular, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            # Cite solution_lesson_node_00126: No Dropout on residual branch
        )

        # 3. Linear Autoregressive Stream
        # Cite solution_lesson_node_00060: Over-parameterize linear stream
        self.linear_proj = nn.Linear(n_linear, latent_dim)

        # 4. Final Head (Shared)
        # Cite solution_lesson_node_00055: Fused representation for both mu and sigma
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(latent_dim, 2))

    def forward(self, images, tabular):
        """
        Args:
            images: Tensor (B, 3, 260, 260)
            tabular: Tensor (B, 5) -> [Base_FVC, Time, Age, Sex, Smoke]
        """
        # --- Stream A: Deep Interaction ---
        # Extract visual features
        x_img = self.backbone.features(images)
        x_img = self.backbone.avgpool(x_img)
        x_img = torch.flatten(x_img, 1)  # (B, 1408)

        # Early Fusion: Concatenate Image + All Tabular
        x_deep_in = torch.cat([x_img, tabular], dim=1)
        latent_deep = self.deep_mlp(x_deep_in)

        # --- Stream B: Linear Autoregressive ---
        # Select Base_FVC (idx 0) and Time (idx 1)
        x_linear_in = tabular[:, :2]
        latent_linear = self.linear_proj(x_linear_in)

        # --- Fusion ---
        # Summation in Latent Space (Cite solution_lesson_node_00052)
        latent_total = latent_deep + latent_linear

        # --- Output ---
        out = self.head(latent_total)

        mu = out[:, 0]
        sigma = F.softplus(out[:, 1]) + 1e-6

        return torch.stack([mu, sigma], dim=1)
