import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ImageEncoder(nn.Module):
    """
    Image branch of MACAN using EfficientNet-B2.
    Unfreezes top layers for domain adaptation.
    """

    def __init__(self):
        super().__init__()
        # Load EfficientNet B2
        # num_classes=0 returns the global pool output (features)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            num_classes=0,
            in_chans=Config.NUM_SLICES,
        )

        # 1. Freeze entire backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze top two convolutional stages (blocks)
        # EfficientNet structure in timm: blocks is a Sequential of blocks
        # We unfreeze the last two blocks
        for param in self.backbone.blocks[-2:].parameters():
            param.requires_grad = True

        # 3. Unfreeze head components (conv_head, bn2) if they exist
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # Projection to shared latent dimension
        self.projection = nn.Linear(self.backbone.num_features, Config.PROJECTION_DIM)

    def forward(self, x):
        # x shape: (Batch, 3, 260, 260)
        features = self.backbone(x)  # Shape: (Batch, num_features)
        projected = self.projection(features)  # Shape: (Batch, 64)
        return projected


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network.
    Stream A: Deep Interaction (Image + Tabular)
    Stream B: Linear Residual (Baseline + Time)
    Cite solution_lesson_node_00052: Dual-Stream Residuals for Strong Autoregressive Signals
    """

    def __init__(self):
        super().__init__()

        # --- Image Branch ---
        self.img_encoder = ImageEncoder()

        # --- Stream A: Deep Interaction ---
        # Input: Image (64) + Tabular (5: Base, Time, Age, Sex, Smoke)
        input_dim_deep = Config.PROJECTION_DIM + 5

        self.stream_a = nn.Sequential(
            nn.Linear(input_dim_deep, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.LATENT_DIM),
        )

        # --- Stream B: Linear Residual ---
        # Input: Base_FVC, Rel_Weeks (Indices 0 and 1 of tabular)
        # Cite solution_lesson_node_00060: Over-Parameterization of Linear Baselines
        input_dim_linear = 2

        self.stream_b = nn.Linear(input_dim_linear, Config.LATENT_DIM)

        # --- Shared Head ---
        self.head = nn.Linear(Config.LATENT_DIM, 2)

    def forward(self, image, tabular):
        """
        Args:
            image: Tensor of shape (B, 3, H, W)
            tabular: Tensor of shape (B, 5)
        """
        # 1. Image Encoding
        img_emb = self.img_encoder(image)  # (B, 64)

        # 2. Stream A: Deep Interaction
        # Concatenate Image + All Tabular
        deep_input = torch.cat([img_emb, tabular], dim=1)
        deep_out = self.stream_a(deep_input)  # (B, 64)

        # 3. Stream B: Linear Residual
        # Extract Base_FVC and Rel_Weeks (Indices 0, 1)
        linear_input = tabular[:, :2]
        linear_out = self.stream_b(linear_input)  # (B, 64)

        # 4. Fusion (Summation)
        # Cite solution_lesson_node_00052: Summing linear and deep streams
        fused = deep_out + linear_out

        # 5. Prediction Head
        out = self.head(fused)  # (B, 2)

        mu = out[:, 0]
        raw_sigma = out[:, 1]

        sigma = F.softplus(raw_sigma) + 1e-6

        return mu, sigma


# Alias for backward compatibility
MACAN = DSPRNet
