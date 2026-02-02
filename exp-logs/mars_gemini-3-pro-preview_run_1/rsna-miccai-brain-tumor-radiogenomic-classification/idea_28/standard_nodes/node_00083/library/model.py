import torch
import torch.nn as nn
import timm
from library.config import Config


class MGMTModel(nn.Module):
    """
    Standard EfficientNet-B0 backbone.
    Uses 3-channel input (FLAIR, T1wCE, T2w) corresponding to the middle slice.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        pretrained=True,
        num_classes=1,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        super(MGMTModel, self).__init__()

        # Load the backbone model using timm
        # Explicitly set in_chans=3 to use standard ImageNet weights without modification
        # Cite solution_lesson_node_00009: Avoid naive channel stacking > 3
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate,
            in_chans=Config.IN_CHANNELS,
        )

    def forward(self, x):
        return self.backbone(x)
