import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ClinicalAnchor(nn.Module):
    """
    Stream A: Over-Parameterized Clinical Anchor.
    Learns the baseline disease trajectory from structured clinical data.
    """

    def __init__(self):
        super(ClinicalAnchor, self).__init__()

        # Inputs: Base_FVC, Relative_Time, Age, Sex, Smoke
        input_dim = Config.CLINICAL_INPUT_DIM
        hidden_dim = Config.CLINICAL_HIDDEN_DIM
        latent_dim = Config.CLINICAL_LATENT_DIM

        # Feature Extractor MLP
        # Linear(Input -> 128) -> ReLU -> Linear(128 -> 64)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),  # Activation before latent usage usually helps, though prompt didn't strictly specify, standard practice for embedding.
            # Prompt says: Linear(128->64). It implies the output of this is H_clin.
        )

        # Base Prediction Head
        # Linear(64 -> 2) -> [mu_base, sigma_base_raw]
        self.head = nn.Linear(latent_dim, 2)

    def forward(self, x):
        # x shape: (Batch, 5)
        h_clin = self.net(x)
        preds = self.head(h_clin)
        return h_clin, preds


class VisualResidual(nn.Module):
    """
    Stream B: Regularized Cascaded Visual Residual.
    Learns a correction (residual) to the clinical baseline using CT scans.
    """

    def __init__(self):
        super(VisualResidual, self).__init__()

        # Backbone: EfficientNet-B2
        # num_classes=0 removes the classifier, returning pooled features if global_pool is set (default in timm)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # Feature dimension of EfficientNet-B2 (usually 1408)
        self.img_feature_dim = self.backbone.num_features

        # Freeze lower layers, unfreeze top layers
        # Freezing all first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the Conv Head and BN
        for param in self.backbone.conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone.bn2.parameters():
            param.requires_grad = True

        # Unfreeze the last 2 blocks (EfficientNet usually has 7 blocks)
        # We access the blocks via .blocks
        num_blocks = len(self.backbone.blocks)
        for i in range(num_blocks - 2, num_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

        # Fusion Dimension: Image Features + Clinical Latent
        fusion_dim = self.img_feature_dim + Config.CLINICAL_LATENT_DIM

        # Residual MLP
        # Linear(Fused -> 128) -> ReLU -> Dropout -> Linear(128 -> 64) -> Dropout -> Linear(64 -> 2)
        self.residual_net = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(128, 64),
            nn.ReLU(),  # Added ReLU to match standard MLP structure between layers
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(64, 2),
        )

        # Zero Initialization of the final layer
        # This ensures the residual starts at 0, so the model starts as the Clinical Anchor
        self._init_zero_output()

    def _init_zero_output(self):
        final_layer = self.residual_net[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, img, h_clin):
        # img shape: (Batch, 3, 260, 260)
        # h_clin shape: (Batch, 64)

        # Extract image features
        img_emb = self.backbone(img)  # (Batch, 1408)

        # Concatenate with clinical latent
        fused = torch.cat([img_emb, h_clin], dim=1)  # (Batch, 1472)

        # Predict residuals
        residuals = self.residual_net(fused)  # (Batch, 2)

        return residuals


class RCOSRNet(nn.Module):
    """
    Regularized Cascaded Output-Space Residual Network.
    Combines Clinical Anchor and Visual Residual via probabilistic output fusion.
    """

    def __init__(self):
        super(RCOSRNet, self).__init__()
        self.clinical_anchor = ClinicalAnchor()
        self.visual_residual = VisualResidual()

    def forward(self, image, clinical):
        """
        Args:
            image (torch.Tensor): CT slices (B, 3, H, W)
            clinical (torch.Tensor): Clinical features (B, 5)
        Returns:
            mu (torch.Tensor): Predicted FVC mean (B, 1)
            sigma (torch.Tensor): Predicted confidence (B, 1)
        """
        # 1. Stream A: Clinical Anchor
        # h_clin: (B, 64), base_preds: (B, 2)
        h_clin, base_preds = self.clinical_anchor(clinical)

        # 2. Stream B: Visual Residual
        # residuals: (B, 2)
        residuals = self.visual_residual(image, h_clin)

        # 3. Output Fusion
        # Split predictions into Mean and Sigma components
        # Index 0: Mean, Index 1: Raw Sigma (Logit)

        base_mu = base_preds[:, 0:1]
        base_sigma_raw = base_preds[:, 1:2]

        res_mu = residuals[:, 0:1]
        res_sigma_raw = residuals[:, 1:2]

        # Additive Residual for Mean
        mu_final = base_mu + res_mu

        # Additive Residual for Sigma (in logit space), then Softplus
        # This allows the image to increase or decrease uncertainty
        sigma_final = F.softplus(base_sigma_raw + res_sigma_raw) + 1e-6

        return mu_final, sigma_final
