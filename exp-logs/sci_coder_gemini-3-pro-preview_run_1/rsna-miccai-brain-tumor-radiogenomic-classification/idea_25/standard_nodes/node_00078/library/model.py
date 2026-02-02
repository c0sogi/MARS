import torch
import torch.nn as nn
import timm
from library import config


class EfficientNet3Channel(nn.Module):
    """
    Standard EfficientNet-B0 taking 3-channel inputs.
    Cite solution_lesson_node_00009: Reverting to 3 channels to avoid destabilization.
    """

    def __init__(self, backbone_name=config.BACKBONE, pretrained=True, num_classes=1):
        super(EfficientNet3Channel, self).__init__()

        # Cite solution_lesson_node_00012: Silent Defaults vs Explicit Overrides
        # We rely on the default dropout (typically 0.2) by not passing drop_rate explicitly
        # or passing the default value.
        # Cite debug_lesson_8: Dynamically Link Model Input Channels to Data Pipeline Configuration
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=config.INPUT_CHANNELS,
        )

    def forward(self, x):
        return self.backbone(x)
