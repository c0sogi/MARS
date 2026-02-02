import torch
import torch.nn as nn
import timm
from library.config import Config


class SETIModel(nn.Module):
    """
    Model for SETI technosignature detection.

    Uses a timm backbone (e.g., ResNet18d) with modified input stem for 6-channel input.
    Channel stacking allows the model to learn comparisons between On/Off targets (Cite Lesson 00003).
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    ):
        super(SETIModel, self).__init__()

        # Create the model using timm.
        # in_chans=6 triggers adaptation of the first layer.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.backbone(x)
