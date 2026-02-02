import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-B2 backbone with specific fine-tuning strategy.
    Freezes lower layers and unfreezes the top two convolutional stages
    and the head to allow domain adaptation while preserving robust low-level features.
    """

    def __init__(self):
        super().__init__()
        # Load pretrained EfficientNet-B2
        # num_classes=0 removes the classifier, returning the pooled feature vector
        # global_pool='avg' ensures we get a flat vector (Batch, Num_Features)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        # Project high-dim features (1408 for B2) to compact latent space
        self.projection = nn.Linear(self.backbone.num_features, Config.PROJECTION_DIM)

        # --- Fine-Tuning Strategy ---
        # 1. Freeze everything initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze the Head (Conv + BN before pooling)
        # In timm efficientnet, this is usually conv_head and bn2
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # 3. Unfreeze the top two convolutional stages (last 2 blocks)
        # self.backbone.blocks is a nn.Sequential of blocks
        if hasattr(self.backbone, "blocks"):
            num_blocks = len(self.backbone.blocks)
            # Unfreeze the last 2 blocks
            for i in range(num_blocks - 2, num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

    def forward(self, x):
        # x: (B, 3, H, W)
        # features: (B, 1408)
        features = self.backbone(x)
        # projected: (B, 64)
        return self.projection(features)


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPRNet).
    Cite Lesson 52: Dual-Stream Residuals for Strong Autoregressive Signals.
    Cite Lesson 60: Over-Parameterization of Linear Baselines.
    """

    def __init__(self):
        super().__init__()

        # 1. Image Branch
        self.encoder = EfficientNetEncoder()

        # 2. Stream A: Linear Trend (Autoregressive)
        # Input: [Baseline_FVC, Time] (Dim=2)
        # Projects strictly linear features to latent space without non-linearities.
        # This preserves the strong linear signal (Lesson 60).
        self.linear_trend = nn.Linear(2, Config.MLP_OUT_DIM)

        # 3. Stream B: Deep Interaction (Residual Correction)
        # Input: Image Projection (64) + All Clinical (5) = 69
        # Models complex non-linear deviations from the trend.
        self.interaction_mlp = nn.Sequential(
            nn.Linear(Config.PROJECTION_DIM + 5, Config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.MLP_HIDDEN_DIM, Config.MLP_OUT_DIM),
        )

        # 4. Zero Initialization for Interaction Stream
        # Forces model to start as a linear regressor (Lesson 52).
        self._init_zero_interaction()

        # 5. Shared Head
        self.head = nn.Linear(Config.MLP_OUT_DIM, 2)

    def _init_zero_interaction(self):
        last_layer = self.interaction_mlp[2]
        if isinstance(last_layer, nn.Linear):
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)

    def forward(self, image, clinical):
        """
        Args:
            image: (B, 3, H, W)
            clinical: (B, 5) -> [Baseline_FVC, Time, Age, Sex, Smoking]
        """
        # --- Stream A: Linear Trend ---
        # Extract Baseline_FVC (idx 0) and Time (idx 1)
        trend_in = clinical[:, :2]
        trend_out = self.linear_trend(trend_in)  # (B, 64)

        # --- Stream B: Deep Interaction ---
        img_embed = self.encoder(image)  # (B, 64)
        inter_in = torch.cat([img_embed, clinical], dim=1)  # (B, 69)
        inter_out = self.interaction_mlp(inter_in)  # (B, 64)

        # --- Latent Summation ---
        # Prediction = LinearTrend + NonLinearCorrection
        h_final = trend_out + inter_out

        # --- Head ---
        logits = self.head(h_final)  # (B, 2)

        mu = logits[:, 0]
        sigma = F.softplus(logits[:, 1]) + 1e-6

        return mu, sigma
