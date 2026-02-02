import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class ModalityAwareEfficientNet(nn.Module):
    """
    Modality-Aware EfficientNet-B0 (Early Fusion).
    Cite solution_lesson_node_00051: Early Fusion vs Siamese.
    """

    def __init__(self):
        super(ModalityAwareEfficientNet, self).__init__()

        # 1. Load Pretrained Backbone
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Modify Stem (First Convolutional Layer)
        # Cite solution_lesson_node_00023: Distribute filters across groups
        original_stem = self.backbone.features[0][0]

        new_stem = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,  # 12
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=4,  # Enforce modality isolation
            bias=False,
        )

        with torch.no_grad():
            new_stem.weight.copy_(original_stem.weight)

        self.backbone.features[0][0] = new_stem

        # 3. Define Classifier Head
        # Cite solution_lesson_node_00017: Include Dropout
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE), nn.Linear(1280, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor (B, 12, 224, 224)
        Returns:
            logits: Tensor (B, 1)
        """
        return self.backbone(x)
