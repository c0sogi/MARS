import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPRNet).
    Fuses a strong linear stream (Baseline+Time) with a deep interaction stream (Image+Meta)
    in the latent space. (Cite Lesson 00052)
    """

    def __init__(self):
        super(DSPRNet, self).__init__()

        # ---------------------------------------------------------------------
        # Deep Stream (Stream A) Components
        # ---------------------------------------------------------------------
        # Backbone: EfficientNet-B2
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # Freeze lower layers, unfreeze top layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze Head and last 2 blocks
        for param in self.backbone.conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone.bn2.parameters():
            param.requires_grad = True

        num_blocks = len(self.backbone.blocks)
        for i in range(num_blocks - 2, num_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

        img_dim = self.backbone.num_features
        clin_dim = Config.CLINICAL_INPUT_DIM
        latent_dim = Config.CLINICAL_LATENT_DIM  # 64

        # Deep Interaction MLP
        # Fuses Image Embeddings + All Clinical Features
        # No Dropout (Cite Lesson 00126)
        self.deep_mlp = nn.Sequential(
            nn.Linear(img_dim + clin_dim, 256), nn.ReLU(), nn.Linear(256, latent_dim)
        )

        # ---------------------------------------------------------------------
        # Linear Stream (Stream B) Components
        # ---------------------------------------------------------------------
        # Projects [Base_FVC, Time] to Latent Space
        # Over-parameterization (Cite Lesson 00060)
        self.linear_proj = nn.Linear(2, latent_dim)

        # ---------------------------------------------------------------------
        # Shared Head
        # ---------------------------------------------------------------------
        self.head = nn.Linear(latent_dim, 2)

    def forward(self, image, clinical):
        """
        Args:
            image (torch.Tensor): CT slices (B, 3, H, W)
            clinical (torch.Tensor): Clinical features (B, 5)
                                     [Base_FVC, Time, Age, Sex, Smoke]
        """
        # 1. Linear Stream (Stream B)
        # Extract Base_FVC and Time (Indices 0 and 1)
        linear_input = clinical[:, :2]
        z_linear = self.linear_proj(linear_input)

        # 2. Deep Stream (Stream A)
        img_emb = self.backbone(image)
        deep_input = torch.cat([img_emb, clinical], dim=1)
        z_deep = self.deep_mlp(deep_input)

        # 3. Latent Fusion (Summation) (Cite Lesson 00052)
        z_fused = z_linear + z_deep

        # 4. Projection
        preds = self.head(z_fused)

        mu = preds[:, 0:1]
        sigma_raw = preds[:, 1:2]
        sigma = F.softplus(sigma_raw) + 1e-6

        return mu, sigma
