import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ClinicalStream(nn.Module):
    """
    Stream A: The Clinical Anchor.
    Processes clinical metadata into a latent vector representing the expected trajectory.
    """

    def __init__(self, input_dim=9, hidden_dim=64, output_dim=64):
        super(ClinicalStream, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class VisualResidualStream(nn.Module):
    """
    Stream B: The Conditional Visual Stream.
    Takes image features and the clinical latent vector to predict a residual correction.
    """

    def __init__(self, img_dim=64, clin_dim=64, hidden_dim=64, output_dim=64):
        super(VisualResidualStream, self).__init__()
        input_dim = img_dim + clin_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, img_embed, clin_embed):
        # Concatenate along feature dimension
        x = torch.cat([img_embed, clin_embed], dim=1)
        return self.net(x)


class CLRNet(nn.Module):
    """
    Cascaded Latent-Residual Network (CLR-Net).

    Architecture:
    1. EfficientNet-B2 Backbone (Top 2 stages unfrozen) -> Image Features
    2. Clinical MLP -> Clinical Latent (H_clin)
    3. Visual MLP(Image + H_clin) -> Residual Latent (H_resid)
    4. H_final = H_clin + H_resid
    5. Head(H_final) -> mu, sigma
    """

    def __init__(self):
        super(CLRNet, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Image Branch (Backbone)
        # ---------------------------------------------------------------------
        # Load pretrained EfficientNet-B2
        # num_classes=0 removes the classifier
        # global_pool='' ensures we get spatial features for custom pooling
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0, global_pool=""
        )

        # Feature dimension for EfficientNet-B2 is 1408
        self.num_features = self.backbone.num_features

        # Freezing Logic
        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze Top Layers
        # EfficientNet structure in timm: conv_stem, bn1, blocks (0-6), conv_head, bn2

        # Unfreeze Head components
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # Unfreeze last 2 blocks of the 'blocks' container
        if hasattr(self.backbone, "blocks"):
            num_blocks = len(self.backbone.blocks)
            # Unfreeze the last 2 blocks (stages)
            for i in range(num_blocks - 2, num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

        # Pooling and Projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.img_project = nn.Linear(self.num_features, Config.LATENT_DIM)

        # ---------------------------------------------------------------------
        # 2. Clinical Stream
        # ---------------------------------------------------------------------
        # Input features: Base_FVC, Percent, Age, Rel_Time, Sex(2), Smoke(3) -> 9
        self.clinical_stream = ClinicalStream(
            input_dim=9, hidden_dim=64, output_dim=Config.LATENT_DIM
        )

        # ---------------------------------------------------------------------
        # 3. Visual Residual Stream
        # ---------------------------------------------------------------------
        self.visual_stream = VisualResidualStream(
            img_dim=Config.LATENT_DIM,
            clin_dim=Config.LATENT_DIM,
            hidden_dim=64,
            output_dim=Config.LATENT_DIM,
        )

        # ---------------------------------------------------------------------
        # 4. Prediction Head
        # ---------------------------------------------------------------------
        self.head = nn.Linear(Config.LATENT_DIM, 2)

        # Dropout for regularization
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, img, clin_data):
        """
        Args:
            img (torch.Tensor): (B, 3, H, W)
            clin_data (torch.Tensor): (B, 9)

        Returns:
            torch.Tensor: (B, 2) -> [mean, raw_sigma]
        """
        # --- Image Path ---
        # Extract features
        features = self.backbone.forward_features(img)  # (B, C, H, W)
        pooled = self.global_pool(features).flatten(1)  # (B, C)
        img_lat = self.img_project(pooled)  # (B, 64)
        img_lat = F.relu(img_lat)
        img_lat = self.dropout(img_lat)

        # --- Clinical Path ---
        clin_lat = self.clinical_stream(clin_data)  # (B, 64)

        # --- Cascaded Residual Path ---
        # Pass processed clinical latent into visual stream along with image latent
        resid_lat = self.visual_stream(img_lat, clin_lat)  # (B, 64)

        # --- Fusion ---
        # H_final = H_clin + H_resid
        # This enforces the residual learning paradigm
        final_lat = clin_lat + resid_lat

        # --- Prediction ---
        out = self.head(final_lat)

        return out
