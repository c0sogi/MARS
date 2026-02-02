import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class ClinicalAnchor(nn.Module):
    """
    Stream A: Over-Parameterized Clinical Anchor.
    Acts as the base learner, predicting the disease trajectory based on clinical metadata.
    """

    def __init__(self):
        super(ClinicalAnchor, self).__init__()

        input_dim = Config.CLINICAL_INPUT_DIM  # 5
        hidden_dims = Config.CLINICAL_HIDDEN_DIMS  # [128, 64]
        output_dim = 2  # Mean (mu) and Raw Uncertainty (sigma_logit)

        # Construct MLP
        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, tabular_input):
        """
        Args:
            tabular_input: Tensor of shape (B, 5)
        Returns:
            Tensor of shape (B, 2) containing [mu_base, sigma_base_raw]
        """
        return self.mlp(tabular_input)


class VisualResidual(nn.Module):
    """
    Stream B: Visual Residual Stream.
    Uses EfficientNet-B2 to learn a residual correction to the clinical anchor.
    Features Output-Space Zero-Initialization.
    """

    def __init__(self):
        super(VisualResidual, self).__init__()

        # 1. Backbone: EfficientNet-B2
        # Weights='DEFAULT' loads the best available pre-trained weights
        weights = models.EfficientNet_B2_Weights.DEFAULT if Config.PRETRAINED else None
        self.backbone = models.efficientnet_b2(weights=weights)

        # 2. Feature Extraction Setup
        # EfficientNet-B2 outputs 1408 channels at the final feature map
        self.feature_dim = 1408

        # Remove the original classifier
        self.backbone.classifier = nn.Identity()

        # 3. Freezing Strategy
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the top two convolutional stages.
        # In torchvision's EfficientNet implementation, 'features' is a Sequential container.
        # B2 has indices 0 to 8. We unfreeze 7 and 8.
        for i in range(7, 9):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True

        # 4. Fusion & Head
        # We concat image features (1408) + clinical features (5)
        fusion_dim = self.feature_dim + Config.CLINICAL_INPUT_DIM

        # Residual MLP Head
        # We use a bottleneck structure before the final projection
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [delta_mu, delta_sigma]
        )

        # 5. Zero Initialization
        # Explicitly initialize the last layer to zero to ensure
        # the residual starts at 0, preserving the Clinical Anchor's stable start.
        self._zero_init_last_layer()

    def _zero_init_last_layer(self):
        last_layer = self.head[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)

    def forward(self, images, tabular_input):
        """
        Args:
            images: Tensor of shape (B, 3, 260, 260)
            tabular_input: Tensor of shape (B, 5)
        Returns:
            Tensor of shape (B, 2) containing [delta_mu, delta_sigma]
        """
        # Extract visual features
        # backbone.features gives (B, 1408, H, W)
        x = self.backbone.features(images)

        # Global Average Pooling -> (B, 1408)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)

        # Early Fusion: Concatenate with clinical data
        fused = torch.cat([x, tabular_input], dim=1)

        # Predict residuals
        residuals = self.head(fused)

        return residuals


class ZIOSRNet(nn.Module):
    """
    Zero-Initialized Output-Space Residual Network.
    Combines Clinical Anchor and Visual Residual streams via summation.
    """

    def __init__(self):
        super(ZIOSRNet, self).__init__()

        self.anchor = ClinicalAnchor()
        self.residual = VisualResidual()

    def forward(self, images, tabular_input):
        """
        Args:
            images: (B, 3, H, W)
            tabular_input: (B, 5)
        Returns:
            fvc_pred: (B,) Predicted FVC (Mean)
            sigma_pred: (B,) Predicted Confidence (Std Dev)
        """
        # Stream A: Base Prediction
        out_anchor = self.anchor(tabular_input)
        mu_base = out_anchor[:, 0]
        sigma_raw_base = out_anchor[:, 1]

        # Stream B: Residual Correction
        out_residual = self.residual(images, tabular_input)
        delta_mu = out_residual[:, 0]
        delta_sigma = out_residual[:, 1]

        # Output-Space Summation
        # 1. Mean: Direct summation
        mu_final = mu_base + delta_mu

        # 2. Uncertainty: Sum logits, then Softplus
        # This allows the visual stream to increase or decrease uncertainty
        # relative to the clinical baseline before enforcing positivity.
        sigma_raw_final = sigma_raw_base + delta_sigma
        sigma_final = F.softplus(sigma_raw_final) + 1e-6  # epsilon for stability

        return mu_final, sigma_final
