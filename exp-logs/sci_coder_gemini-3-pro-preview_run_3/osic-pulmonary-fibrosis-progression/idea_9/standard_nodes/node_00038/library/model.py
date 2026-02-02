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


class CAPNet(nn.Module):
    """
    Context-Aware Point-Wise Network (CAP-Net).
    Predicts FVC and Uncertainty for a specific time point using fused features.
    Cite solution_lesson_node_00037: Point-Wise Conditioning vs. Parametric Trajectory Prediction.
    """

    def __init__(self):
        super(CAPNet, self).__init__()

        # 1. Image Branch
        self.image_encoder = ImageEncoder(projection_dim=Config.PROJECTION_DIM)

        # 2. Fusion & MLP
        # Input: Image Projection (128) + Tabular Features (4) + Time (1)
        # Cite solution_lesson_node_00037: Include Time as an explicit input feature.
        input_dim = Config.PROJECTION_DIM + Config.N_TABULAR_FEATURES + 1

        # Cite solution_lesson_node_00027: Increase capacity (Hidden Dimension: 256 -> 512)
        hidden_dim = 512

        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 3. Output Head
        # Predicts mu and sigma directly
        self.head = nn.Linear(hidden_dim, 2)

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
        # Concatenate image embeddings, raw tabular features, and time
        # Cite solution_lesson_node_00037: Feed Time into dense layers
        combined = torch.cat([img_embed, tabular, t_rel], dim=1)

        # 3. Shared Representation
        hidden = self.fusion_mlp(combined)

        # 4. Predict Mu and Sigma
        out = self.head(hidden)

        mu = out[:, 0:1]

        # Cite solution_lesson_node_00014: Do not enforce domain-specific clipping in architecture
        # Use simple softplus + epsilon for stability
        sigma = F.softplus(out[:, 1:2]) + 1e-6

        return mu, sigma
