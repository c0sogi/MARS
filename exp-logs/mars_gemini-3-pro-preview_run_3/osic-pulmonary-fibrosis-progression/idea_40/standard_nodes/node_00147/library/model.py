import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class ClinicalStream(nn.Module):
    """
    Stream A: Over-Parameterized Clinical Anchor.
    Processes tabular data to estimate baseline trajectory.
    """

    def __init__(self):
        super().__init__()
        # Input: Baseline FVC, Relative Time, Age, Sex, SmokingStatus (5 features)
        self.net = nn.Sequential(
            nn.Linear(Config.N_TABULAR_FEATURES, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Linear(64, Config.OUTPUT_DIM),  # Outputs: [Mean_Base, Logit_Sigma_Base]
        )

    def forward(self, x):
        return self.net(x)


class VisualStream(nn.Module):
    """
    Stream B: Context-Injected Visual Residual.
    Uses EfficientNet-B2 to extract visual features, fused with context (Base FVC, Time),
    to predict residuals for mean and uncertainty.
    """

    def __init__(self):
        super().__init__()

        # Load Backbone
        # We use the default pre-trained weights (ImageNet)
        weights = models.EfficientNet_B2_Weights.DEFAULT
        self.backbone = models.efficientnet_b2(weights=weights)

        # Feature dimension for EfficientNet-B2 is 1408
        self.feature_dim = self.backbone.classifier[1].in_features

        # Freezing Logic
        # 1. Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze top two stages (last two blocks of the features sequential container)
        # This allows the network to adapt high-level semantic features while keeping low-level detectors fixed
        for param in self.backbone.features[-1].parameters():
            param.requires_grad = True
        for param in self.backbone.features[-2].parameters():
            param.requires_grad = True

        # Bottleneck Projection
        # Project high-dimensional image features to a compact embedding to balance dimensionality
        # with the low-dimensional context features (Cite solution_lesson_node_00146).
        self.projection_dim = 128
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.projection_dim), nn.ReLU()
        )

        # Context Injection Input Dimension
        # Projected Image Features (128) + Raw Baseline FVC (1) + Relative Time (1)
        input_dim = self.projection_dim + Config.N_CONTEXT_FEATURES

        # Residual MLP
        # Explicitly excluding Dropout as per Idea to preserve weak residual signals
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Linear(
                64, Config.OUTPUT_DIM
            ),  # Outputs: [Mean_Residual, Logit_Sigma_Residual]
        )

    def forward(self, images, context):
        # Extract visual features
        # (B, 3, H, W) -> (B, 1408, H', W')
        x = self.backbone.features(images)

        # Global Average Pooling -> (B, 1408, 1, 1)
        x = self.backbone.avgpool(x)

        # Flatten -> (B, 1408)
        x = torch.flatten(x, 1)

        # Project to lower dimension (Bottleneck)
        x = self.projector(x)

        # Inject Context (Concatenation)
        # x: (B, 128), context: (B, 2) -> (B, 130)
        x = torch.cat([x, context], dim=1)

        # Predict Residuals
        return self.mlp(x)


class CIDSNet(nn.Module):
    """
    Context-Injected Dual-Stream Network.
    Fuses a clinical anchor stream with a visual residual stream.
    """

    def __init__(self):
        super().__init__()
        self.clinical_stream = ClinicalStream()
        self.visual_stream = VisualStream()

    def forward(self, images, tabular):
        """
        Args:
            images: Tensor (B, 3, 260, 260)
            tabular: Tensor (B, 5) -> [Base_FVC, Time, Age, Sex, Smoke]
        """
        # 1. Stream A: Clinical Anchor
        # Uses all tabular features to establish the baseline trajectory
        out_a = self.clinical_stream(tabular)
        mu_base = out_a[:, 0]
        sigma_logit_base = out_a[:, 1]

        # 2. Stream B: Visual Residual
        # Extract context: Base_FVC_Scaled (idx 0) and Time_Scaled (idx 1)
        # These provide the necessary patient-specific context to the visual stream
        context = tabular[:, :2]
        out_b = self.visual_stream(images, context)
        mu_res = out_b[:, 0]
        sigma_logit_res = out_b[:, 1]

        # 3. Fusion
        # Mean: Additive Residual (Base + Correction)
        mu_final = mu_base + mu_res

        # Uncertainty: Additive Logits -> Softplus
        # This allows the visual stream to increase or decrease uncertainty relative to the baseline
        # We add a small epsilon to ensure numerical stability
        sigma_final = F.softplus(sigma_logit_base + sigma_logit_res) + 1e-6

        # Return stacked output (B, 2) -> [FVC_Prediction, Confidence]
        return torch.stack([mu_final, sigma_final], dim=1)
