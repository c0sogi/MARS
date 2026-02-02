import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ClinicalAnchor(nn.Module):
    """
    Stream A: Over-Parameterized Clinical Anchor.
    Learns the baseline disease trajectory from clinical scalars.
    """

    def __init__(self, input_dim=5, hidden_dim=128):
        super().__init__()
        # Architecture: Linear(Input -> 128) -> ReLU -> Linear(128 -> 64) -> Linear(64 -> 2)
        # Note: As per design, the 64-dim layer acts as a linear bottleneck before the final projection.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 5) containing [Baseline_FVC, t_rel, Age, Sex, Smoking]
        Returns:
            Tensor of shape (B, 2) containing [Base_Mean, Base_Sigma_Logit]
        """
        return self.net(x)


class VisualResidual(nn.Module):
    """
    Stream B: Regularized Visual Residual.
    Extracts features from CT scans to predict a residual correction to the anchor.
    """

    def __init__(
        self,
        backbone_name="efficientnet_b2",
        pretrained=True,
        clinical_dim=5,
        proj_dim=64,
        hidden_dim=128,
        drop_rate=0.2,
    ):
        super().__init__()

        # 1. Backbone (EfficientNet-B2)
        # num_classes=0 ensures we get the pooled feature vector (B, num_features)
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )

        # 2. Freezing Logic
        # Freeze all parameters initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze top two stages (Head + Last 2 Blocks)
        # This allows the model to adapt high-level features while preserving low-level filters.
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # Unfreeze the last 2 blocks of the 'blocks' sequence
        if hasattr(self.backbone, "blocks"):
            for block in self.backbone.blocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True

        # 3. Bottleneck Projection
        # Projects high-dim image features (1408) to low-dim (64) to prevent noise dominance
        n_features = self.backbone.num_features
        self.projection = nn.Linear(n_features, proj_dim)

        # 4. Residual MLP
        # Architecture: Linear(Fused -> 128) -> ReLU -> Dropout -> Linear(128 -> 64) -> Dropout -> Linear(64 -> 2)
        fusion_dim = proj_dim + clinical_dim
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, 64),
            nn.Dropout(drop_rate),
            nn.Linear(64, 2),
        )

        # 5. Zero Initialization
        # Initialize the final layer to zero so the model starts exactly as the Clinical Anchor
        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, images, clinical):
        """
        Args:
            images: Tensor of shape (B, 3, 260, 260)
            clinical: Tensor of shape (B, 5)
        Returns:
            Tensor of shape (B, 2) containing [Delta_Mean, Delta_Sigma_Logit]
        """
        # Extract features
        features = self.backbone(images)  # (B, 1408)

        # Project to bottleneck
        img_proj = self.projection(features)  # (B, 64)

        # Context Injection: Concatenate with clinical scalars
        combined = torch.cat([img_proj, clinical], dim=1)  # (B, 69)

        # Predict residuals
        return self.mlp(combined)


class BCOSRNet(nn.Module):
    """
    Boundary-Constrained Output-Space Residual Network.
    Fuses the Clinical Anchor and Visual Residual streams with architectural constraints.
    """

    def __init__(self):
        super().__init__()

        # Stream A
        self.anchor = ClinicalAnchor(input_dim=5, hidden_dim=Config.HIDDEN_DIM)

        # Stream B
        self.residual = VisualResidual(
            backbone_name=Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            clinical_dim=5,
            proj_dim=Config.PROJECTION_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            drop_rate=Config.DROP_RATE,
        )

        # Pre-calculate normalized sigma floor
        # The loss function unscales predictions by multiplying by Config.TARGET_STD.
        # To enforce output * STD >= 70, we enforce output >= 70 / STD.
        self.sigma_floor_norm = Config.SIGMA_FLOOR / Config.TARGET_STD

    def forward(self, images, clinical):
        """
        Args:
            images: (B, 3, H, W)
            clinical: (B, 5)
        Returns:
            Tensor of shape (B, 2): [Mean_Normalized, Sigma_Normalized]
        """
        # 1. Stream A: Anchor Predictions
        anchor_out = self.anchor(clinical)
        mu_base = anchor_out[:, 0]
        sigma_base_logit = anchor_out[:, 1]

        # 2. Stream B: Residual Predictions
        res_out = self.residual(images, clinical)
        delta_mu = res_out[:, 0]
        delta_sigma_logit = res_out[:, 1]

        # 3. Fusion & Constraints

        # Mean: Additive residual
        mu_final = mu_base + delta_mu

        # Uncertainty: Sum logits -> Softplus -> Add Floor
        # This ensures gradients flow through the softplus even for low uncertainty,
        # and strictly enforces the minimum bound required by the metric.
        sigma_total_logit = sigma_base_logit + delta_sigma_logit
        sigma_final = F.softplus(sigma_total_logit) + self.sigma_floor_norm

        # Stack for output
        return torch.stack([mu_final, sigma_final], dim=1)
