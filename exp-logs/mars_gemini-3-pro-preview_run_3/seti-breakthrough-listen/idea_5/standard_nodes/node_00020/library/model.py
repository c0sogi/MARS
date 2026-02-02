import torch
import torch.nn as nn
import timm
from library.utils import Config


class SiameseSpatialFusionNet(nn.Module):
    """
    Siamese Difference Network (Cite Lesson 00019).

    Architecture:
    1. Shared Backbone: EfficientNet-B0 (ImageNet weights) extracts global feature vectors.
    2. Explicit Difference: v_diff = v_on - v_off.
    3. Classification Head: Linear layer on difference vector.
    """

    def __init__(self):
        super(SiameseSpatialFusionNet, self).__init__()

        # 1. Siamese Backbone
        # Use global_pool='avg' to get feature vectors (N, C)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        if hasattr(self.backbone, "num_features"):
            self.backbone_dim = self.backbone.num_features
        else:
            self.backbone_dim = 1280

        # 2. Classification Head
        # Input is the difference vector
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(self.backbone_dim, 1)
        )

    def forward(self, x_on, x_off):
        # Pass through shared backbone to get global vectors
        v_on = self.backbone(x_on)  # Shape: (N, C)
        v_off = self.backbone(x_off)  # Shape: (N, C)

        # Explicit Feature Differencing (Cite Lesson 00019)
        # This hardcodes the subtraction logic required to isolate the signal
        v_diff = v_on - v_off

        # Classification
        logits = self.classifier(v_diff)  # Shape: (N, 1)

        return logits
