import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPR-Net).

    Cite Lesson 00052: Dual-Stream Residuals for Strong Autoregressive Signals.
    Cite Lesson 00060: Over-Parameterization of Linear Baselines.

    A dual-stream architecture that explicitly separates the linear autoregressive
    signal (Baseline FVC + Time) from the complex non-linear image interactions.
    """

    def __init__(self):
        super(DSPRNet, self).__init__()

        # --- Image Backbone (Fine-Tuned Content-Adaptive 2.5D) ---
        # EfficientNet-B2
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, in_chans=3, features_only=False
        )

        # Determine backbone output dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            features = self.backbone.forward_features(dummy)
            self.backbone_dim = features.shape[1]

        # Pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Projection to lower dimension
        self.img_project = nn.Linear(self.backbone_dim, Config.IMG_EMBED_DIM)

        # --- Freezing Strategy ---
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze top two convolutional stages and head
        for name, child in self.backbone.named_children():
            if name == "blocks":
                for i, block in enumerate(child):
                    if i >= 5:
                        for param in block.parameters():
                            param.requires_grad = True
            elif name in ["conv_head", "bn2"]:
                for param in child.parameters():
                    param.requires_grad = True

        # --- Stream A: Linear Residual Stream (Cite Lesson 00052, 00060) ---
        # Input: [Base_FVC, Base_Pct, Rel_Time, Age, Sex, Smoke] (6 features)
        # We use an over-parameterized linear mapping to the latent space
        # to avoid bottlenecking the strong linear signal.
        self.clinical_input_dim = 6
        self.linear_stream = nn.Linear(self.clinical_input_dim, Config.LATENT_DIM)

        # --- Stream B: Deep Interaction Stream ---
        # Input: Image Projection + Clinical Features
        self.visual_input_dim = Config.IMG_EMBED_DIM + self.clinical_input_dim
        self.deep_stream = nn.Sequential(
            nn.Linear(self.visual_input_dim, Config.VISUAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.VISUAL_HIDDEN_DIM, Config.LATENT_DIM),
        )

        # --- Shared Head ---
        # Projects fused latent to Mu and Sigma
        self.head = nn.Linear(Config.LATENT_DIM, 2)

    def forward(self, image, tabular):
        """
        Args:
            image: Tensor (Batch, 3, H, W)
            tabular: Tensor (Batch, 6)
        Returns:
            Tensor (Batch, 2) -> [FVC_Mean, Confidence_Sigma]
        """
        # 1. Image Feature Extraction
        x = self.backbone.forward_features(image)  # (B, C, H', W')
        x = self.global_pool(x).flatten(1)  # (B, C)
        img_embed = self.img_project(x)  # (B, 64)

        # 2. Stream A (Linear Residual)
        h_lin = self.linear_stream(tabular)  # (B, 64)

        # 3. Stream B (Deep Interaction)
        vis_input = torch.cat([img_embed, tabular], dim=1)
        h_deep = self.deep_stream(vis_input)  # (B, 64)

        # 4. Residual Fusion (Summation)
        # Combines the robust linear trend with the deep correction
        h_final = h_lin + h_deep

        # 5. Prediction Head
        out = self.head(h_final)

        mu = out[:, 0].unsqueeze(1)
        raw_sigma = out[:, 1].unsqueeze(1)

        # Constraint: Sigma must be positive.
        # Cite Lesson 00014: Softplus + epsilon (no hard clip in architecture)
        sigma = F.softplus(raw_sigma) + 1e-6

        return torch.cat([mu, sigma], dim=1)
