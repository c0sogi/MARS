import torch
import torch.nn as nn
import timm
from library.config import Config


class VisualStream(nn.Module):
    """
    Stream B Part 1: Feature Extraction and Projection.
    Uses EfficientNet-B2 and projects high-dim features to low-dim space.
    Cite Lesson 146: Apply projection layer to high-dim stream before fusion.
    """

    def __init__(self, backbone_name, projection_dim):
        super().__init__()
        # Load backbone with pooled output (num_classes=0 returns global pool features)
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, in_chans=3
        )

        # Determine backbone output dimension (EfficientNet-B2 is typically 1408)
        self.num_features = self.backbone.num_features

        # Bottleneck Projection: 1408 -> 64
        self.projection = nn.Linear(self.num_features, projection_dim)

        # Freeze bottom layers, unfreeze top two stages
        self._freeze_layers()

    def _freeze_layers(self):
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze top stages (blocks 5, 6 for B2) and head components
        # We look for specific strings in parameter names corresponding to the top blocks
        targets = ["blocks.5", "blocks.6", "conv_head", "bn2"]
        for name, param in self.backbone.named_parameters():
            if any(t in name for t in targets):
                param.requires_grad = True

    def forward(self, x):
        # Extract features (B, 1408)
        x = self.backbone(x)
        # Project (B, 64)
        x = self.projection(x)
        return x


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network.
    Cite Lesson 52: Use a linear residual stream for strong baseline dependency.
    Cite Lesson 60: Over-parameterize linear stream (project to latent) before fusion.
    Cite Lesson 139: Critical state variables (Baseline FVC) visible to all branches.
    """

    def __init__(self):
        super().__init__()

        # Configuration
        self.tabular_input_dim = 5  # [fvc_norm, age_norm, sex, smoke, time]
        self.hidden_dim = Config.HIDDEN_DIM
        self.feature_dim = Config.PROJECTION_DIM

        # Stream A: Linear Residual (Autoregressive Baseline)
        # Maps clinical features directly to latent space linearly
        self.linear_stream = nn.Linear(self.tabular_input_dim, self.feature_dim)

        # Stream B: Deep Interaction (Visual + Clinical)
        self.visual_stream = VisualStream(
            backbone_name=Config.BACKBONE_NAME, projection_dim=self.feature_dim
        )

        # Deep Interaction MLP
        # Input: Projected Visual (64) + Raw Clinical (5) = 69
        self.interaction_mlp = nn.Sequential(
            nn.Linear(self.feature_dim + self.tabular_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )

        # Shared Head: Feature Dim -> 2 (mu, sigma)
        self.head = nn.Linear(self.feature_dim, 2)

    def forward(self, image, tabular):
        """
        Args:
            image: Tensor (B, 3, H, W)
            tabular: Tensor (B, 5)
        Returns:
            final_out: Tensor (B, 2) [mu, sigma]
        """
        # 1. Stream A: Linear Residual
        # h_linear: (B, 64)
        h_linear = self.linear_stream(tabular)

        # 2. Stream B: Deep Interaction
        # h_vis_proj: (B, 64)
        h_vis_proj = self.visual_stream(image)

        # Context Injection (Cite Lesson 139)
        # (B, 64) + (B, 5) -> (B, 69)
        context_input = torch.cat([h_vis_proj, tabular], dim=1)

        # h_deep: (B, 64)
        h_deep = self.interaction_mlp(context_input)

        # 3. Latent Fusion
        # Summing the streams (Cite Lesson 52)
        h_final = h_linear + h_deep

        # 4. Prediction
        # final_out: (B, 2)
        final_out = self.head(h_final)

        return final_out
