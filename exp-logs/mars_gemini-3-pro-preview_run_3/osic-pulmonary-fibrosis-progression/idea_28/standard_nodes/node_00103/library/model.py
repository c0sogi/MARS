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
    Cite solution_lesson_node_00052: Dual-Stream Residuals.
    Cite solution_lesson_node_00060: Over-Parameterization of Linear Baselines.
    """

    def __init__(self):
        super().__init__()

        # --- Image Branch ---
        self.img_encoder = ImageEncoder()

        # --- Stream A: Deep Interaction Stream ---
        # Inputs: Image (64) + Tabular (8)
        # Tabular: [Base_FVC, Rel_Weeks, Age, Sex_M, Sex_F, Smoke_Ex, Smoke_N, Smoke_C]
        self.tab_encoder = nn.Sequential(
            nn.Linear(8, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.PROJECTION_DIM),  # 64
        )

        # Fusion of Image + Deep Tabular
        self.deep_fusion = nn.Sequential(
            nn.Linear(Config.PROJECTION_DIM * 2, Config.HIDDEN_DIM),  # 128 -> 128
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.LATENT_DIM),  # 128 -> 64
        )

        # --- Stream B: Linear Residual Stream ---
        # Cite solution_lesson_node_00060: Project linear baseline to latent dim.
        # Inputs: Scaled_Base_FVC (idx 0), Scaled_Rel_Weeks (idx 1)
        self.linear_stream = nn.Linear(2, Config.LATENT_DIM)

        # --- Shared Head ---
        # Cite solution_lesson_node_00055: Uncertainty from shared representation.
        self.head = nn.Linear(Config.LATENT_DIM, 2)

    def forward(self, image, tabular):
        """
        Args:
            image: Tensor of shape (B, 3, H, W)
            tabular: Tensor of shape (B, 8)
        Returns:
            mu: Predicted FVC (Z-scored scale)
            sigma: Predicted Confidence (Z-scored scale, positive)
        """
        # 1. Deep Stream
        img_emb = self.img_encoder(image)  # (B, 64)
        tab_emb = self.tab_encoder(tabular)  # (B, 64)

        deep_in = torch.cat([img_emb, tab_emb], dim=1)  # (B, 128)
        deep_emb = self.deep_fusion(deep_in)  # (B, 64)

        # 2. Linear Stream (Residual)
        # Extract Base_FVC and Rel_Weeks
        linear_in = tabular[:, :2]
        linear_emb = self.linear_stream(linear_in)  # (B, 64)

        # 3. Summation in Latent Space
        # Cite solution_lesson_node_00052
        final_emb = deep_emb + linear_emb  # (B, 64)

        # 4. Prediction
        out = self.head(final_emb)
        mu = out[:, 0]
        sigma = F.softplus(out[:, 1]) + 1e-6

        return mu, sigma
