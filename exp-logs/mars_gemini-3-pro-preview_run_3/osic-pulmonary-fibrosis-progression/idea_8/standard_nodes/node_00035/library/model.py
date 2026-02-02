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


class TabularEncoder(nn.Module):
    """
    Tabular branch processing clinical metadata.
    Explicitly excludes Batch/Layer Norm to preserve magnitude signals of Baseline FVC.
    Input: (B, 7) -> [Base_FVC, Base_Percent, Base_Age, Sex, Smoke_Ex, Smoke_Never, Smoke_Current]
    Output: (B, EMBEDDING_DIM)
    """

    def __init__(self, input_dim=7, embedding_dim=EMBEDDING_DIM):
        super(TabularEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class TAPNet(nn.Module):
    """
    Time-Aware Parametric Network (TAP-Net).
    Fuses image and tabular features to predict trajectory parameters.
    """

    def __init__(self):
        super(TAPNet, self).__init__()
        self.image_encoder = EfficientNetEncoder()
        # Tabular input dim is 7 based on data.py preprocessing
        self.tabular_encoder = TabularEncoder(input_dim=7)

        # Fusion Head
        # Concatenates Image (128) + Tabular (128) -> 256
        self.head = nn.Sequential(
            nn.Linear(EMBEDDING_DIM * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, image, tabular):
        """
        Args:
            image: (B, 3, H, W)
            tabular: (B, 7)

        Returns:
            params: (B, 4) containing raw parameters:
                0: alpha (Baseline Autoregressive Coefficient)
                1: beta (Intercept Adjustment)
                2: gamma (Rate of Decline/Slope)
                3: delta (Uncertainty parameter)
        """
        img_emb = self.image_encoder(image)
        tab_emb = self.tabular_encoder(tabular)

        # Concatenate embeddings
        combined = torch.cat([img_emb, tab_emb], dim=1)

        # Predict parameters
        params = self.head(combined)

        return params
