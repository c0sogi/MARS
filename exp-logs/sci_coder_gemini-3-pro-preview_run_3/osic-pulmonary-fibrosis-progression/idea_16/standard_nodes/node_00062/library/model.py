import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class RaliNet(nn.Module):
    """
    Residual-Augmented Latent Interaction Network (RALI-Net).

    Architecture:
    1. Backbone: EfficientNet-B2 (Fine-tuned top stages).
    2. Stream A: Over-Parameterized Clinical MLP (Models trajectory).
    3. Stream B: Visual Interaction MLP (Models visual residuals).
    4. Fusion: Summation of Stream A and Stream B latents.
    """

    def __init__(self):
        super(RaliNet, self).__init__()

        # ==============================
        # 1. Image Backbone (EfficientNet-B2)
        # ==============================
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            num_classes=0,  # Remove classifier
            global_pool="",  # We will handle pooling manually
        )

        # Feature dimension for EfficientNet-B2 is typically 1408
        self.num_features = self.backbone.num_features

        # Freezing Strategy:
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze top layers (conv_head, bn2, and last 2 blocks)
        # Note: timm EfficientNet structure usually has 'blocks', 'conv_head', 'bn2'
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True

        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # Unfreeze last 2 blocks in the 'blocks' container
        # EfficientNet-B2 has 7 stages in 'blocks'
        if hasattr(self.backbone, "blocks"):
            num_blocks = len(self.backbone.blocks)
            # Unfreeze the last 2 blocks
            for i in range(num_blocks - 2, num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

        # Image Projection Layer
        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_projector = nn.Linear(self.num_features, Config.VISUAL_PROJECTION_DIM)

        # ==============================
        # 2. Stream A: Clinical Residual
        # ==============================
        # Input: 7 Clinical Features
        # Architecture: Linear -> ReLU -> Linear
        self.clinical_net = nn.Sequential(
            nn.Linear(Config.CLINICAL_INPUT_DIM, Config.CLINICAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.CLINICAL_HIDDEN_DIM, Config.LATENT_DIM),
        )

        # ==============================
        # 3. Stream B: Visual Interaction
        # ==============================
        # Input: Visual Projection (128) + Clinical (7)
        # Architecture: Linear -> ReLU -> Linear
        self.interaction_net = nn.Sequential(
            nn.Linear(Config.INTERACTION_INPUT_DIM, Config.INTERACTION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.INTERACTION_HIDDEN_DIM, Config.LATENT_DIM),
        )

        # ==============================
        # 4. Prediction Head
        # ==============================
        # Input: Fused Latent (128)
        # Output: 2 (FVC, Raw Sigma)
        self.head = nn.Linear(Config.LATENT_DIM, 2)

    def forward(self, images, tabular):
        """
        Args:
            images (torch.Tensor): (Batch, 3, H, W)
            tabular (torch.Tensor): (Batch, 7)
        Returns:
            torch.Tensor: (Batch, 2) -> [FVC, Raw_Sigma]
        """
        # --- Image Branch ---
        # Extract features: (B, C, H, W)
        features = self.backbone.forward_features(images)
        # Pool: (B, C, 1, 1) -> (B, C)
        pooled = self.img_pool(features).flatten(1)
        # Project: (B, 128)
        visual_latent = self.img_projector(pooled)

        # --- Stream A: Clinical Residual ---
        # Models the robust clinical trajectory
        # (B, 7) -> (B, 128)
        clinical_latent = self.clinical_net(tabular)

        # --- Stream B: Visual Interaction ---
        # Concatenate Visual and Clinical features
        # (B, 128) + (B, 7) -> (B, 135)
        combined_input = torch.cat([visual_latent, tabular], dim=1)
        # Models the non-linear interaction / residual correction
        # (B, 135) -> (B, 128)
        interaction_latent = self.interaction_net(combined_input)

        # --- Latent Fusion ---
        # Summation forces Stream B to learn a residual to Stream A
        # (B, 128)
        fused_latent = clinical_latent + interaction_latent

        # --- Prediction ---
        # (B, 2)
        output = self.head(fused_latent)

        return output
