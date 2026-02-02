import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPRNet).
    Fuses a strong linear autoregressive stream with a deep interaction stream in latent space.
    Cite Lesson 00052, 00060.
    """

    def __init__(self):
        super(DSPRNet, self).__init__()

        # ---------------------------------------------------------------------
        # Stream A: Linear Trend (Wide)
        # ---------------------------------------------------------------------
        # Inputs: Base_FVC, Time
        # Cite Lesson 00060: Over-parameterize linear stream (2 -> 64)
        self.linear_stream = nn.Linear(2, 64)

        # ---------------------------------------------------------------------
        # Stream B: Deep Interaction (Deep)
        # ---------------------------------------------------------------------
        # 1. Image Backbone (EfficientNet-B2)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # Unfreeze top layers (Differential Learning Rate strategy)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone.bn2.parameters():
            param.requires_grad = True
        num_blocks = len(self.backbone.blocks)
        for i in range(num_blocks - 2, num_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

        # 2. Clinical Encoder
        # Inputs: Base_FVC, Time, Age, Sex, Smoke
        self.clinical_encoder = nn.Sequential(
            nn.Linear(Config.CLINICAL_INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # 3. Deep Fusion (Image + Clinical)
        fusion_dim = self.backbone.num_features + 64
        self.deep_fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
        )
        # Cite Lesson 00126: Avoid stochastic regularization (Dropout) on residual branch

        # ---------------------------------------------------------------------
        # Shared Head
        # ---------------------------------------------------------------------
        # Fuses Stream A and Stream B latents
        self.head = nn.Linear(64, 2)

    def forward(self, image, clinical):
        """
        Args:
            image (torch.Tensor): CT slices (B, 3, H, W)
            clinical (torch.Tensor): Clinical features (B, 5)
        """
        # Stream A: Linear Trend
        # clinical[:, :2] corresponds to [Base_FVC_Std, Time_Scaled]
        h_linear = self.linear_stream(clinical[:, :2])

        # Stream B: Deep Interaction
        img_emb = self.backbone(image)
        clin_emb = self.clinical_encoder(clinical)
        fused_deep = torch.cat([img_emb, clin_emb], dim=1)
        h_deep = self.deep_fusion(fused_deep)

        # Latent Summation (Cite Lesson 00052)
        h_final = h_linear + h_deep

        # Prediction
        preds = self.head(h_final)

        mu = preds[:, 0:1]
        sigma = F.softplus(preds[:, 1:2]) + 1e-6

        return mu, sigma
