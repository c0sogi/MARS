import torch
import torch.nn as nn
import timm
from library.config import EMBEDDING_DIM


class EfficientNetEncoder(nn.Module):
    """
    Image branch using EfficientNet-B1 to extract deep semantic features.
    Input: (B, 3, H, W) - 3 slices treated as RGB channels.
    Output: (B, EMBEDDING_DIM)
    """

    def __init__(
        self, model_name="efficientnet_b1", pretrained=True, embedding_dim=EMBEDDING_DIM
    ):
        super(EfficientNetEncoder, self).__init__()
        # Load EfficientNet-B1. num_classes=0 returns the pooled feature vector (Global Average Pooling).
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        in_features = self.backbone.num_features

        # Projection layer to shared embedding space
        self.project = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        # x: (Batch, 3, Height, Width)
        features = self.backbone(x)  # (Batch, num_features)
        out = self.project(features)  # (Batch, embedding_dim)
        return out


class TAPNet(nn.Module):
    """
    Time-Aware Parametric Network (TAP-Net).
    Fuses image and tabular features to predict trajectory parameters.
    Cite solution_lesson_node_00035: Direct Concatenation of Strong Scalar Predictors.
    """

    def __init__(self):
        super(TAPNet, self).__init__()
        self.image_encoder = EfficientNetEncoder()
        # Tabular input dim is 7 based on data.py preprocessing
        # We remove the TabularEncoder MLP to preserve strong scalar signals

        # Fusion Head
        # Concatenates Image (128) + Tabular (7) -> 135
        self.head = nn.Sequential(
            nn.Linear(EMBEDDING_DIM + 7, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
        )

    def forward(self, image, tabular):
        """
        Args:
            image: (B, 3, H, W)
            tabular: (B, 7)

        Returns:
            params: (B, 5) containing raw parameters:
                0: alpha (Baseline Autoregressive Coefficient)
                1: beta (Intercept Adjustment)
                2: gamma (Rate of Decline/Slope)
                3: delta_base (Baseline Uncertainty)
                4: delta_growth (Uncertainty Growth Rate)
        """
        img_emb = self.image_encoder(image)

        # Concatenate image embedding with raw tabular features
        # Cite solution_lesson_node_00035
        combined = torch.cat([img_emb, tabular], dim=1)

        # Predict parameters
        params = self.head(combined)

        return params
