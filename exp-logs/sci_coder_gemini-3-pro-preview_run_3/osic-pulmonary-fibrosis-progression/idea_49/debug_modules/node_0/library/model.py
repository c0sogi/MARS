import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ImageBranch(nn.Module):
    """
    Fine-Tuned Content-Adaptive 2.5D Image Branch.
    Extracts features from CT scans and projects them to a low-dimensional bottleneck.
    """

    def __init__(self):
        super(ImageBranch, self).__init__()
        # Load EfficientNet-B2
        # num_classes=0 removes the classifier and returns pooled features (1408 dim for B2)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.BACKBONE_PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # --- Fine-Tuning Strategy ---
        # 1. Freeze all parameters initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze the top two convolutional stages
        # EfficientNet-B2 structure in timm typically has 'blocks' 0 through 6.
        # We unfreeze the last two blocks (5 and 6) and the final conv head/bn.
        if hasattr(self.backbone, "blocks"):
            for param in self.backbone.blocks[5].parameters():
                param.requires_grad = True
            for param in self.backbone.blocks[6].parameters():
                param.requires_grad = True

        # Unfreeze conv_head and bn2 (final layers before pooling)
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # 3. Bottleneck Projection
        # Projects high-dimensional image noise (1408) to structural embedding (64)
        self.projection = nn.Linear(Config.BACKBONE_OUT_DIM, Config.PROJECTION_DIM)

    def forward(self, x):
        # x: (B, 3, H, W)
        features = self.backbone(x)  # (B, 1408)
        projected = self.projection(features)  # (B, 64)
        return projected


class StreamA(nn.Module):
    """
    Stream A: Over-Parameterized Clinical Anchor.
    Learns the baseline disease trajectory from clinical scalars.
    """

    def __init__(self, input_dim=5):
        super(StreamA, self).__init__()
        # MLP: Input -> 128 -> ReLU -> 64 -> 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, 64),
            nn.Linear(64, Config.OUTPUT_DIM),  # Outputs: mu_base, sigma_base_raw
        )

    def forward(self, x):
        return self.net(x)


class StreamB(nn.Module):
    """
    Stream B: Context-Injected Visual Residual.
    Learns patient-specific corrections (residuals) based on image and clinical context.
    """

    def __init__(self, input_dim):
        super(StreamB, self).__init__()
        # Input dim = Image Projection (64) + Clinical (5) = 69
        # MLP: Input -> 128 -> ReLU -> 64 -> 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, 64),
            nn.Linear(64, Config.OUTPUT_DIM),  # Outputs: delta_mu, delta_sigma_raw
        )
        # Note: Using standard PyTorch initialization (Kaiming Uniform).
        # Explicitly avoiding Zero-Initialization to prevent gradient stagnation.

    def forward(self, x):
        return self.net(x)


class SCARNet(nn.Module):
    """
    Standardized Constraint-Aware Residual Network.
    Combines a clinical anchor with a visual residual stream, enforcing metric constraints architecturally.
    """

    def __init__(self):
        super(SCARNet, self).__init__()

        # 1. Image Branch
        self.image_branch = ImageBranch()

        # Clinical features dimension is 5:
        # [Baseline_FVC_Scaled, Time_Scaled, Age_Scaled, Sex_Code, Smoking_Code]
        self.clinical_dim = 5

        # 2. Stream A (Anchor)
        self.stream_a = StreamA(input_dim=self.clinical_dim)

        # 3. Stream B (Residual)
        # Input: Image Projection (64) + Clinical (5)
        self.stream_b_input_dim = Config.PROJECTION_DIM + self.clinical_dim
        self.stream_b = StreamB(input_dim=self.stream_b_input_dim)

        # Metric Constraint Floor (Standardized)
        self.sigma_floor = Config.SIGMA_FLOOR_STD

    def forward(self, images, tabular):
        """
        Args:
            images (torch.Tensor): Batch of images (B, 3, H, W)
            tabular (torch.Tensor): Batch of clinical features (B, 5)
        Returns:
            preds (torch.Tensor): (B, 2) -> [mu_scaled, sigma_scaled]
        """
        # 1. Extract Image Features
        img_features = self.image_branch(images)  # (B, 64)

        # 2. Stream A: Predict Baseline Trajectory
        base_out = self.stream_a(tabular)  # (B, 2)
        mu_base = base_out[:, 0:1]
        sigma_base_raw = base_out[:, 1:2]

        # 3. Stream B: Predict Residuals
        # Context Injection: Concatenate projected image features with raw clinical scalars
        combined_features = torch.cat([img_features, tabular], dim=1)  # (B, 69)
        residual_out = self.stream_b(combined_features)  # (B, 2)
        delta_mu = residual_out[:, 0:1]
        delta_sigma_raw = residual_out[:, 1:2]

        # 4. Constraint-Aware Fusion

        # Mean: Base + Residual
        mu_scaled = mu_base + delta_mu

        # Uncertainty: Softplus(Base + Residual) + Floor
        # We sum the raw logits first to allow the residual to freely correct the base,
        # then apply Softplus to ensure positivity, then add the architectural floor.
        sigma_combined_raw = sigma_base_raw + delta_sigma_raw
        sigma_scaled = F.softplus(sigma_combined_raw) + self.sigma_floor

        # Stack outputs
        preds = torch.cat([mu_scaled, sigma_scaled], dim=1)

        return preds
