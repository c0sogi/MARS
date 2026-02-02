import torch
import torch.nn as nn
import timm
from library.config import (
    MODEL_NAME,
    PRETRAINED,
    DROP_RATE,
    DROP_PATH_RATE,
    TAB_HIDDEN_DIM,
    NUM_CLASSES,
)


class HybridEfficientNet(nn.Module):
    """
    Hybrid Vision-Tabular architecture combining EfficientNet-B3 with a metadata MLP.

    Attributes:
        backbone (nn.Module): EfficientNet-B3 feature extractor.
        tabular_mlp (nn.Sequential): MLP for processing tabular metadata.
        classifier (nn.Linear): Final fully connected layer for fusion and prediction.
    """

    def __init__(self, meta_dim, pretrained=PRETRAINED):
        """
        Args:
            meta_dim (int): The number of input features in the tabular metadata.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(HybridEfficientNet, self).__init__()

        # ====================================================
        # Visual Branch: EfficientNet-B3
        # ====================================================
        # num_classes=0 removes the original classification head.
        # global_pool='avg' ensures the output is a pooled 1D vector (Batch, Num_Features).
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="gem",
            drop_rate=DROP_RATE,
            drop_path_rate=DROP_PATH_RATE,
        )

        # Feature dimension for EfficientNet-B3 is typically 1536
        self.vis_feature_dim = self.backbone.num_features

        # ====================================================
        # Tabular Branch: Lightweight MLP
        # ====================================================
        # Processes the one-hot encoded and normalized metadata.
        # We use a simple 2-layer MLP with BatchNorm and Dropout for regularization.
        self.tabular_mlp = nn.Sequential(
            nn.Linear(meta_dim, TAB_HIDDEN_DIM),
            nn.BatchNorm1d(TAB_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(TAB_HIDDEN_DIM, TAB_HIDDEN_DIM),
            nn.BatchNorm1d(TAB_HIDDEN_DIM),
            nn.ReLU(),
        )

        # ====================================================
        # Fusion Head
        # ====================================================
        # Concatenates visual and tabular features.
        fusion_dim = self.vis_feature_dim + TAB_HIDDEN_DIM

        # Output layer: Projects to NUM_CLASSES (1 for binary classification)
        self.classifier = nn.Linear(fusion_dim, NUM_CLASSES)

    def forward(self, images, meta):
        """
        Forward pass of the hybrid model.

        Args:
            images (torch.Tensor): Batch of images (B, C, H, W).
            meta (torch.Tensor): Batch of metadata features (B, meta_dim).

        Returns:
            torch.Tensor: Logits (B, NUM_CLASSES).
        """
        # 1. Extract Visual Features
        # Shape: (Batch, 1536)
        vis_features = self.backbone(images)

        # 2. Extract Tabular Features
        # Shape: (Batch, TAB_HIDDEN_DIM)
        tab_features = self.tabular_mlp(meta)

        # 3. Feature Fusion
        # Concatenate along the feature dimension
        # Shape: (Batch, 1536 + TAB_HIDDEN_DIM)
        combined_features = torch.cat([vis_features, tab_features], dim=1)

        # 4. Final Prediction
        # Shape: (Batch, 1)
        logits = self.classifier(combined_features)

        return logits
