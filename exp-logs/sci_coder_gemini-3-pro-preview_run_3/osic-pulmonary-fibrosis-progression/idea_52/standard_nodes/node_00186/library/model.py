import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class AILRNet(nn.Module):
    """
    Affine-Isolated Latent-Residual Network (AILR-Net).

    A hybrid CNN-MLP architecture utilizing a Parallel Dual-Stream Latent Summation topology.
    It explicitly resolves the conflict between signal preservation and capacity by implementing
    a restricted-input, over-parameterized anchor fused with a context-aware visual residual.
    """

    def __init__(self):
        super(AILRNet, self).__init__()

        # --- Stream A: Affine-Isolated Clinical Anchor ---
        # Input: [Baseline_FVC_scaled, Relative_Time] (Dim: 2)
        # Purpose: Learn the strong autoregressive identity mapping and linear decay.
        # Structure: Over-Parameterized MLP (2 -> 128 -> 128)
        self.stream_a = nn.Sequential(
            nn.Linear(2, Config.LATENT_DIM, bias=True),
            nn.ReLU(),
            nn.Linear(Config.LATENT_DIM, Config.LATENT_DIM, bias=True),
        )

        # --- Stream B: Context-Aware Visual Residual ---
        # Backbone: EfficientNet-B2
        # Input: Images (3, 260, 260)
        self.backbone = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.DEFAULT
        )

        # Feature Extraction Dimension for EfficientNet-B2 is 1408
        self.n_features = self.backbone.classifier[1].in_features

        # Remove the original classifier to get raw features
        self.backbone.classifier = nn.Identity()

        # Freezing Strategy: Freeze bottom, unfreeze top two stages
        # "features" in torchvision EfficientNet contains blocks 0..8
        # We freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last two blocks (indices 7 and 8 for B2)
        for param in self.backbone.features[-2:].parameters():
            param.requires_grad = True

        # Context Fusion MLP
        # Input: Image Features (1408) + Full Context (5)
        # Fix: Project Image Features (1408 -> 128) then Fuse (128 + 5)
        # Cite solution_lesson_node_00146: Avoid concatenating vectors with orders-of-magnitude differences.
        self.img_projector = nn.Linear(self.n_features, 128, bias=True)

        self.context_fusion_dim = 128 + 5
        self.stream_b_mlp = nn.Sequential(
            nn.Linear(self.context_fusion_dim, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, Config.LATENT_DIM, bias=True),
        )

        # --- Head: Latent Fusion & Constraint-Aware Output ---
        # Shared head for mu and sigma
        self.head = nn.Linear(Config.LATENT_DIM, 2, bias=True)

        # --- Metric Constraint Constants ---
        # Calculate epsilon in standardized space: 70 / sigma_global
        self.sigma_epsilon = Config.SIGMA_MIN / Config.TARGET_STD

    def forward(self, images, restricted_inputs, context_inputs):
        """
        Args:
            images (torch.Tensor): (B, 3, H, W)
            restricted_inputs (torch.Tensor): (B, 2) [Base_FVC, Rel_Time]
            context_inputs (torch.Tensor): (B, 5) [Base_FVC, Rel_Time, Age, Sex, Smoke]

        Returns:
            mu (torch.Tensor): Predicted mean (standardized).
            sigma (torch.Tensor): Predicted std (standardized).
        """

        # --- Stream A: Clinical Anchor ---
        # Projects restricted inputs to latent space
        h_a = self.stream_a(restricted_inputs)

        # --- Stream B: Visual Residual ---
        # 1. Extract Image Features
        # EfficientNet forward returns the pooled features if classifier is Identity?
        # Actually torchvision implementation with Identity classifier returns (B, 1408)
        # But we need to be careful about pooling.
        # The .features() returns (B, C, H, W). The .avgpool() returns (B, C, 1, 1).
        # The .classifier() (now Identity) handles the rest.
        # Let's manually call features and pool to be safe and explicit.
        x = self.backbone.features(images)  # (B, 1408, H', W')
        x = self.backbone.avgpool(x)  # (B, 1408, 1, 1)
        x = torch.flatten(x, 1)  # (B, 1408)

        # 2. Context Injection
        # Project high-dim image features to low-dim space first
        x_proj = self.img_projector(x)  # (B, 128)

        # Concatenate projected image features with full clinical context
        fused_input = torch.cat([x_proj, context_inputs], dim=1)  # (B, 128 + 5)

        # 3. Project to latent space
        h_b = self.stream_b_mlp(fused_input)

        # --- Latent Fusion ---
        # Summation enforces residual learning: H_final = Anchor + Residual
        h_final = h_a + h_b

        # --- Output Head ---
        outputs = self.head(h_final)
        mu = outputs[:, 0]
        sigma_raw = outputs[:, 1]

        # --- Architectural Metric Constraint ---
        # Enforce floor: sigma = Softplus(raw) + epsilon
        # This ensures sigma >= epsilon (approx 70ml in scaled space)
        sigma = F.softplus(sigma_raw) + self.sigma_epsilon

        return mu, sigma
