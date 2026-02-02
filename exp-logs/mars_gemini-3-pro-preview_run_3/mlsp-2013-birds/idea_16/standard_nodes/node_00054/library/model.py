import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdClassifier(nn.Module):
    """
    Bird Species Classifier architecture.
    Uses native timm heads to preserve architecture-specific regularization.
    Cite solution_lesson_node_00052: Avoid stripping native heads.
    Cite solution_lesson_node_00053: Avoid aggressive regularization (MSD) on small datasets.
    """

    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool = True):
        super(BirdClassifier, self).__init__()

        # Initialize backbone using timm with native head
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=Config.CHANNELS,
        )

    def forward(self, x):
        return self.model(x)
