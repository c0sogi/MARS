import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config
from library.utils import get_global_stats


class RestrictedAnchorMLP(nn.Module):
    """
    Stream A: Restricted Affine Anchor.
    Receives only [Baseline FVC (scaled), Relative Time].
    Explicitly excludes static metadata to enforce autoregressive bias.
    """

    def __init__(self):
        super().__init__()
        # Input dim: 2
        # Architecture: Linear(2 -> 128) -> ReLU -> Linear(128 -> 64)
        # Bias is strictly True to allow learning global intercepts
        self.net = nn.Sequential(
            nn.Linear(2, Config.LATENT_DIM, bias=True),
            nn.ReLU(),
            nn.Linear(Config.LATENT_DIM, 64, bias=True),
        )

    def forward(self, x):
        return self.net(x)


class VisualContextResidual(nn.Module):
    """
    Stream B: Context-Aware Visual Residual.
    Backbone: EfficientNet-B2 (Unfrozen top stages).
    Inputs: Image + [Base FVC, Rel Time, Age, Sex, Smoking].
    """

    def __init__(self):
        super().__init__()

        # Load Backbone
        # We use 3 input channels as we stack 3 slices (Anchor + 2 neighbors)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            in_chans=3,
            num_classes=0,
            global_pool="avg",
        )

        # Freezing Strategy: Unfreeze top two stages; keep bottom frozen.
        # EfficientNet-B2 structure typically involves blocks.0 through blocks.6.
        # We unfreeze blocks.5, blocks.6, and the head components.
        trainable_keywords = ["blocks.5", "blocks.6", "conv_head", "bn2"]

        for name, param in self.backbone.named_parameters():
            should_train = any(k in name for k in trainable_keywords)
            param.requires_grad = should_train

        self.num_features = self.backbone.num_features

        # Bottleneck Projection for Image Features
        self.img_proj = nn.Linear(self.num_features, Config.IMG_PROJ_DIM)

        # Context Injection
        # Context inputs: 5 dims
        # Input to MLP = IMG_PROJ_DIM (64) + 5 = 69
        mlp_in_dim = Config.IMG_PROJ_DIM + 5

        # Residual MLP (No Dropout)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, Config.LATENT_DIM),
            nn.ReLU(),
            nn.Linear(Config.LATENT_DIM, 64),
        )

    def forward(self, images, context):
        # images: (B, 3, H, W)
        # context: (B, 5)

        # 1. Extract Visual Features
        features = self.backbone(images)  # (B, num_features)

        # 2. Project to lower dimension
        img_proj = self.img_proj(features)  # (B, 64)

        # 3. Concatenate with Context
        combined = torch.cat([img_proj, context], dim=1)  # (B, 69)

        # 4. Compute Residual Latent
        out = self.mlp(combined)  # (B, 64)

        return out


class ARLRNet(nn.Module):
    """
    Affine-Restricted Latent-Residual Network.
    Combines Stream A (Anchor) and Stream B (Residual) via summation.
    Enforces metric constraints architecturally.
    """

    def __init__(self, global_std_target=None):
        super().__init__()

        self.stream_a = RestrictedAnchorMLP()
        self.stream_b = VisualContextResidual()

        # Shared Head
        # Projects fused latent (64) to [mu_scaled, sigma_raw]
        self.head = nn.Linear(64, 2)

        # Metric Constraint setup
        if global_std_target is None:
            # If not provided, calculate from training data
            _, global_std_target = get_global_stats()

        self.global_std_target = global_std_target

        # Calculate epsilon_std to enforce the 70ml floor in scaled space
        # sigma_scaled_min = 70 / sigma_global
        self.epsilon_std = Config.SIGMA_MIN / self.global_std_target

    def forward(self, image, stream_a_input, stream_b_input):
        """
        Args:
            image: (B, 3, H, W)
            stream_a_input: (B, 2) -> [Scaled Base FVC, Rel Time]
            stream_b_input: (B, 5) -> [Scaled Base FVC, Rel Time, Age, Sex, Smoke]
        """
        # 1. Stream A: Anchor
        h_a = self.stream_a(stream_a_input)  # (B, 64)

        # 2. Stream B: Residual
        h_b = self.stream_b(image, stream_b_input)  # (B, 64)

        # 3. Latent Fusion (Summation)
        h_final = h_a + h_b  # (B, 64)

        # 4. Head Projection
        out = self.head(h_final)
        mu_scaled = out[:, 0]
        sigma_raw = out[:, 1]

        # 5. Constraint-Aware Output
        # Enforce sigma >= epsilon_std using Softplus + Offset
        sigma_scaled = F.softplus(sigma_raw) + self.epsilon_std

        # Return stacked output (B, 2)
        return torch.stack([mu_scaled, sigma_scaled], dim=1)
