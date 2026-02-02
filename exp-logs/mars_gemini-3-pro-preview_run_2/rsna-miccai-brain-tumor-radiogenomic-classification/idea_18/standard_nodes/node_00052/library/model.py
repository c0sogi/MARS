import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class ModalityAwareEfficientNet(nn.Module):
    """
    EfficientNet-B0 with Modality-Aware Grouped Convolutional Stem.
    Cite solution_lesson_node_00051: Single-View Early Fusion.
    """

    def __init__(self):
        super(ModalityAwareEfficientNet, self).__init__()

        # 1. Load Pretrained Backbone
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Modify Stem for 12 Channels (Groups=4)
        original_stem = self.backbone.features[0][0]

        new_stem = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,  # 12
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=4,  # Cite solution_lesson_node_00023: Preserving Filter Diversity
            bias=False,
        )

        # 3. Initialize Stem Weights
        # Copy 32 filters directly. Shape matches exactly due to groups=4.
        with torch.no_grad():
            new_stem.weight.copy_(original_stem.weight)

        self.backbone.features[0][0] = new_stem

        # 4. Modify Classifier
        # Preserve Dropout (Cite solution_lesson_node_00017)
        # Original classifier is Sequential(Dropout, Linear(1280, 1000))
        # We replace the Linear layer but keep the Dropout
        self.backbone.classifier[1] = nn.Linear(1280, 1)

    def forward(self, x):
        return self.backbone(x)
