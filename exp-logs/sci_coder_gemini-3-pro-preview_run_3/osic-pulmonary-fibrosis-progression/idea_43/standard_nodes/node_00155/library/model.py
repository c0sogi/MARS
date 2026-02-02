import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPRNet).
    Cite Lesson 52, 60, 146.

    Architecture:
    1. Backbone: EfficientNet-B2 (Fine-tuned).
    2. Stream A (Linear Residual): Linear(Base, Time) -> Latent.
    3. Stream B (Deep Interaction): MLP(Image + Tabular) -> Latent.
    4. Fusion: Latent Summation -> Head.
    """

    def __init__(self):
        super(DSPRNet, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Image Backbone (EfficientNet-B2)
        # ---------------------------------------------------------------------
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="",
        )
        self.backbone_dim = self.backbone.num_features

        # Fine-Tuning Strategy (Cite Lesson 27)
        for param in self.backbone.parameters():
            param.requires_grad = False
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True
        num_blocks = len(self.backbone.blocks)
        blocks_to_unfreeze = 2
        for i in range(max(0, num_blocks - blocks_to_unfreeze), num_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Image Projection (Cite Lesson 146)
        self.img_projector = nn.Linear(self.backbone_dim, Config.BOTTLENECK_DIM)

        # ---------------------------------------------------------------------
        # 2. Stream A: Linear Residual (Latent Projection)
        # ---------------------------------------------------------------------
        # Input: Baseline_FVC, Time (2 features)
        # Project to Latent Space (Cite Lesson 60)
        self.linear_stream = nn.Linear(2, Config.BOTTLENECK_DIM)

        # ---------------------------------------------------------------------
        # 3. Stream B: Deep Interaction (Image + Tabular)
        # ---------------------------------------------------------------------
        # Input: Image Latent (128) + Tabular (5)
        input_dim_deep = Config.BOTTLENECK_DIM + len(Config.TABULAR_FEATURES)

        self.deep_stream = nn.Sequential(
            nn.Linear(input_dim_deep, Config.HIDDEN_DIM),
            nn.ReLU(),
            # No Dropout (Cite Lesson 126)
            nn.Linear(Config.HIDDEN_DIM, Config.BOTTLENECK_DIM),
        )

        # ---------------------------------------------------------------------
        # 4. Shared Head
        # ---------------------------------------------------------------------
        self.head = nn.Linear(Config.BOTTLENECK_DIM, Config.OUTPUT_DIM)

        # Initialization: Standard (Cite Lesson 118)

    def forward(self, image, tabular):
        # Image Features
        features = self.backbone.forward_features(image)
        pooled = self.global_pool(features).flatten(1)
        img_lat = self.img_projector(pooled)

        # Stream A: Linear Residual (Baseline FVC, Time)
        # Tabular indices: 0=Baseline_FVC, 1=Time
        lin_lat = self.linear_stream(tabular[:, :2])

        # Stream B: Deep Interaction
        combined = torch.cat([img_lat, tabular], dim=1)
        deep_lat = self.deep_stream(combined)

        # Fusion: Latent Summation (Cite Lesson 52)
        latent = deep_lat + lin_lat

        # Output
        out = self.head(latent)
        final_mean = out[:, 0]
        final_sigma = F.softplus(out[:, 1]) + 1e-6

        return {
            "final_mean": final_mean,
            "final_sigma": final_sigma,
        }
