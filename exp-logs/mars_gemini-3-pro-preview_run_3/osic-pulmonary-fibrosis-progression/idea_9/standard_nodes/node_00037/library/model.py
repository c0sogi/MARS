import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class ImageEncoder(nn.Module):
    """
    EfficientNet-B0 backbone for processing CT slices.
    Unfreezes top layers for domain adaptation while keeping lower layers frozen.
    """

    def __init__(self, projection_dim=128):
        super(ImageEncoder, self).__init__()

        # Load pretrained EfficientNet-B0
        # Using the default weights (IMAGENET1K_V1)
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)

        # Feature extraction logic:
        # efficientnet_b0.features contains the convolutional blocks.
        # We want to freeze the early layers and unfreeze the top ones.
        # Structure: features[0]...features[8]

        # 1. Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze top blocks (e.g., last 2 blocks and the final conv)
        # features[7] and features[8] are the deeper layers
        for param in self.backbone.features[7].parameters():
            param.requires_grad = True
        for param in self.backbone.features[8].parameters():
            param.requires_grad = True

        # The classifier is not used, but we need a projection layer
        # EfficientNet-B0 output channels before classifier is 1280
        self.num_features = 1280
        self.projection = nn.Linear(self.num_features, projection_dim)

    def forward(self, x):
        # x shape: (Batch, 3, 224, 224)

        # Extract features
        x = self.backbone.features(x)  # (Batch, 1280, 7, 7)

        # Global Average Pooling
        x = self.backbone.avgpool(x)  # (Batch, 1280, 1, 1)
        x = torch.flatten(x, 1)  # (Batch, 1280)

        # Projection
        x = self.projection(x)  # (Batch, 128)

        return x


class OCPNet(nn.Module):
    """
    Origin-Corrected Parametric Network.
    Predicts patient-specific linear trajectory parameters and dynamic uncertainty.
    """

    def __init__(self):
        super(OCPNet, self).__init__()

        # 1. Image Branch
        self.image_encoder = ImageEncoder(projection_dim=Config.PROJECTION_DIM)

        # 2. Fusion & MLP
        # Input: Image Projection (128) + Tabular Features (4)
        input_dim = Config.PROJECTION_DIM + Config.N_TABULAR_FEATURES
        hidden_dim = 64

        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 3. Parametric Heads
        # Trajectory Head: alpha, beta, gamma
        self.traj_head = nn.Linear(hidden_dim, 3)

        # Uncertainty Head: delta_base, delta_growth
        self.unc_head = nn.Linear(hidden_dim, 2)

    def forward(self, images, tabular, t_rel):
        """
        Args:
            images: (Batch, 3, H, W)
            tabular: (Batch, 4) -> [Base_FVC_Norm, Age_Norm, Sex, Smoke]
            t_rel: (Batch, 1) -> Relative time scaled

        Returns:
            mu: (Batch, 1) -> Predicted normalized FVC mean
            sigma: (Batch, 1) -> Predicted normalized FVC uncertainty
        """
        # 1. Image Features
        img_embed = self.image_encoder(images)  # (Batch, 128)

        # 2. Fusion
        # Concatenate image embeddings and raw tabular features
        # Tabular features are NOT passed through BN/LN to preserve magnitude info
        combined = torch.cat([img_embed, tabular], dim=1)  # (Batch, 132)

        # 3. Shared Representation
        hidden = self.fusion_mlp(combined)  # (Batch, 64)

        # 4. Predict Parameters
        traj_params = self.traj_head(hidden)  # (Batch, 3)
        unc_params = self.unc_head(hidden)  # (Batch, 2)

        # Extract specific parameters
        alpha = traj_params[:, 0:1]  # Autoregressive coefficient
        beta = traj_params[:, 1:2]  # Offset
        gamma = traj_params[:, 2:3]  # Slope

        delta_base = unc_params[:, 0:1]
        delta_growth = unc_params[:, 1:2]

        # 5. Calculate Trajectory (Mean)
        # mu(t) = alpha * Base + beta + gamma * t_rel
        # Note: Base FVC is the first column of tabular input
        base_fvc_norm = tabular[:, 0:1]
        mu = alpha * base_fvc_norm + beta + gamma * t_rel

        # 6. Calculate Uncertainty (Sigma)
        # sigma(t) = softplus(delta_base) + softplus(delta_growth) * |t_rel|
        # Enforces positive uncertainty that grows (or changes) with time distance
        sigma = F.softplus(delta_base) + F.softplus(delta_growth) * torch.abs(t_rel)

        return mu, sigma
