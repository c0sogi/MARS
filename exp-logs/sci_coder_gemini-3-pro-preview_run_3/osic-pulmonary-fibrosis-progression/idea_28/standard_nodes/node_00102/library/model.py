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


class MACAN(nn.Module):
    """
    Metric-Aligned Cascaded Anchor Network.
    Combines a clinical anchor MLP with a cascaded visual modifier.
    """

    def __init__(self):
        super().__init__()

        # --- Image Branch ---
        self.img_encoder = ImageEncoder()

        # --- Stream A: Clinical Anchor ---
        # Input features: Scaled_Base_FVC, Scaled_Rel_Weeks, Scaled_Age,
        #                 Sex_Male, Sex_Female, Smoke_Ex, Smoke_Never, Smoke_Current
        # Total input dim = 8
        input_dim_tabular = 8

        self.stream_a = nn.Sequential(
            nn.Linear(input_dim_tabular, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.LATENT_DIM),
        )

        # --- Stream B: Cascaded Visual Interaction ---
        # Input: Concatenation of Image Projection (64) + Stream A Output (64)
        cascade_input_dim = Config.PROJECTION_DIM + Config.LATENT_DIM

        self.stream_b = nn.Sequential(
            nn.Linear(cascade_input_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.LATENT_DIM),
        )

        # --- Shared Head ---
        # Projects the fused latent vector to mu and raw_sigma
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
        # 1. Image Encoding
        img_emb = self.img_encoder(image)  # (B, 64)

        # 2. Stream A (Clinical Anchor)
        clin_emb = self.stream_a(tabular)  # (B, 64)

        # 3. Stream B (Cascade)
        # Concatenate image features with the clinical anchor
        combined = torch.cat([img_emb, clin_emb], dim=1)  # (B, 128)
        vis_correction = self.stream_b(combined)  # (B, 64)

        # 4. Residual Fusion
        # The visual stream acts as a residual correction to the clinical anchor
        final_emb = clin_emb + vis_correction  # (B, 64)

        # 5. Prediction Head
        out = self.head(final_emb)  # (B, 2)

        mu = out[:, 0]
        raw_sigma = out[:, 1]

        # 6. Uncertainty Constraint
        # Apply softplus to ensure positivity
        sigma = F.softplus(raw_sigma) + 1e-6

        return mu, sigma
