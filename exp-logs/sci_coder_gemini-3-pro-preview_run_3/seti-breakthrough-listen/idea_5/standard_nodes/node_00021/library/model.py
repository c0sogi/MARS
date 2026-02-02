import torch
import torch.nn as nn
import timm
from library.utils import Config


class SiameseSpatialFusionNet(nn.Module):
    """
    Siamese Network with Explicit Feature Differencing.

    Architecture:
    1. Shared Backbone: EfficientNet-B0 (ImageNet weights) extracts global feature vectors.
    2. Explicit Difference: Computes (On - Off) in latent space.
    3. Context Concatenation: Feeds [On, Off, Diff] to the classifier.
    4. Classification Head: MLP.
    """

    def __init__(self):
        super(SiameseSpatialFusionNet, self).__init__()

        # 1. Siamese Backbone
        # Load EfficientNet-B0 with ImageNet weights.
        # Use global_pool='avg' to get feature vectors (N, C)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # Determine backbone output channels (1280 for EfficientNet-B0)
        if hasattr(self.backbone, "num_features"):
            self.backbone_dim = self.backbone.num_features
        else:
            self.backbone_dim = 1280

        # 2. Classification Head
        # Input: Concatenation of [u, v, u-v] -> 3 * backbone_dim
        # Cite solution_lesson_node_00020: Preserve context by concatenating original vectors
        input_dim = self.backbone_dim * 3

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x_on, x_off):
        """
        Forward pass of the Siamese Network.
        """
        # Pass through shared backbone to get global feature vectors
        v_on = self.backbone(x_on)  # Shape: (N, 1280)
        v_off = self.backbone(x_off)  # Shape: (N, 1280)

        # Explicit Difference (Cite solution_lesson_node_00019)
        v_diff = v_on - v_off

        # Context Preservation: Concatenate [u, v, u-v]
        v_cat = torch.cat([v_on, v_off, v_diff], dim=1)  # Shape: (N, 3840)

        # Classification
        logits = self.classifier(v_cat)

        return logits
