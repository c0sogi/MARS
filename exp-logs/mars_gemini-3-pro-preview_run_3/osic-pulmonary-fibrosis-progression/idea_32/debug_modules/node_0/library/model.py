import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ClinicalAnchor(nn.Module):
    """
    Stream A: Over-Parameterized Clinical Anchor (The Prior).
    Learns the baseline disease trajectory from clinical metadata.
    """

    def __init__(self, input_dim=5):
        super(ClinicalAnchor, self).__init__()
        # Over-Parameterized MLP: Input -> 128 -> ReLU -> 64 -> ReLU -> 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Outputs: mu_clin, sigma_clin_raw
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tabular features of shape (B, 5).
        Returns:
            torch.Tensor: Shape (B, 2) containing raw mu and raw sigma logits.
        """
        return self.net(x)


class VisualResidual(nn.Module):
    """
    Stream B: Visual Residual Stream (The Correction).
    Uses EfficientNet-B2 to predict corrections to the clinical anchor.
    """

    def __init__(self, input_dim=5):
        super(VisualResidual, self).__init__()

        # 1. Image Backbone
        # efficientnet_b2, num_classes=0 removes the classifier and returns pooled features
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # 2. Fine-Tuning Strategy: Unfreeze top two stages; keep bottom frozen.
        # First, freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the Head components (Conv + BN) if they exist
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # Unfreeze the last 2 blocks (stages)
        # self.backbone.blocks is a nn.Sequential of blocks in timm
        if hasattr(self.backbone, "blocks"):
            num_blocks = len(self.backbone.blocks)
            # Unfreeze last 2 blocks
            for i in range(max(0, num_blocks - 2), num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

        # 3. Fusion & Head
        self.img_feature_dim = self.backbone.num_features
        fusion_dim = self.img_feature_dim + input_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(Config.DROP_RATE),
            nn.Linear(128, 2),  # Outputs: delta_mu, delta_sigma
        )

        # 4. Structural Innovation: Zero Initialization
        self.zero_init_head()

    def zero_init_head(self):
        """
        Explicitly initializes the weights and biases of the final Linear layer to zero.
        This forces the visual stream to output exactly 0 correction at epoch 0.
        """
        final_layer = self.head[-1]
        nn.init.constant_(final_layer.weight, 0.0)
        nn.init.constant_(final_layer.bias, 0.0)

    def forward(self, img, tab):
        """
        Args:
            img (torch.Tensor): Image tensor (B, C, H, W).
            tab (torch.Tensor): Tabular tensor (B, 5).
        Returns:
            torch.Tensor: Residual corrections (B, 2).
        """
        # Extract image features
        img_feats = self.backbone(img)  # (B, num_features)

        # Early Fusion: Concatenate
        fused = torch.cat([img_feats, tab], dim=1)

        # Predict residuals
        residuals = self.head(fused)

        return residuals


class OSPRNet(nn.Module):
    """
    Output-Space Probabilistic Residual Network.
    Synthesizes Dual-Stream Output-Space Summation.
    """

    def __init__(self):
        super(OSPRNet, self).__init__()
        self.clinical_anchor = ClinicalAnchor(input_dim=5)
        self.visual_residual = VisualResidual(input_dim=5)

    def forward(self, img, tab):
        """
        Args:
            img (torch.Tensor): Image tensor (B, 3, 260, 260).
            tab (torch.Tensor): Tabular tensor (B, 5).
        Returns:
            mu (torch.Tensor): Predicted FVC mean (B,).
            sigma (torch.Tensor): Predicted FVC uncertainty (B,).
        """
        # 1. Stream A: Clinical Anchor
        # Returns (B, 2) -> [mu_clin, sigma_clin_raw]
        clin_out = self.clinical_anchor(tab)

        # 2. Stream B: Visual Residual
        # Returns (B, 2) -> [delta_mu, delta_sigma]
        res_out = self.visual_residual(img, tab)

        # 3. Output-Space Summation

        # Mean: mu_final = mu_clin + delta_mu
        mu = clin_out[:, 0] + res_out[:, 0]

        # Uncertainty: sigma_final = Softplus(sigma_clin_raw + delta_sigma) + epsilon
        # We sum the logits to allow the residual to push uncertainty up or down in logit space
        sigma_logit = clin_out[:, 1] + res_out[:, 1]
        sigma = F.softplus(sigma_logit) + 1e-6

        return mu, sigma
