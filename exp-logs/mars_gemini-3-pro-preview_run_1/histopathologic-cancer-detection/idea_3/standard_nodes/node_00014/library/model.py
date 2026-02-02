import torch
import torch.nn as nn
import timm
from library.config import Config


class ConvNeXtTinyCustom(nn.Module):
    """
    ConvNeXt-Tiny architecture adapted for small resolution histopathology patches.

    Cite solution_lesson_node_00008: Partial Transfer Learning with Stem Modification.
    Cite solution_lesson_node_00003: Adapt Standard Backbone Stems for Small Input Resolutions.
    """

    def __init__(self, pretrained: bool = Config.PRETRAINED):
        super().__init__()

        # Load ConvNeXt backbone
        # num_classes=0 returns the pooled feature vector (global pool + flatten)
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=pretrained, num_classes=0
        )

        # ---------------------------------------------------------------------
        # Stem Modification
        # ---------------------------------------------------------------------
        # Original ConvNeXt Stem:
        # Conv2d(3, 96, kernel_size=4, stride=4)
        # This reduces 48x48 -> 12x12 immediately, which is too aggressive.

        # We modify the stem conv to stride 1 to preserve resolution.
        # ConvNeXt stem is usually accessible via model.stem (Sequential)
        # stem[0] is the Conv2d layer.

        if hasattr(self.backbone, "stem"):
            original_conv = self.backbone.stem[0]
            new_conv = nn.Conv2d(
                in_channels=original_conv.in_channels,
                out_channels=original_conv.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=original_conv.bias is not None,
            )
            self.backbone.stem[0] = new_conv

        # ---------------------------------------------------------------------
        # Classifier Head
        # ---------------------------------------------------------------------
        self.num_features = self.backbone.num_features
        self.fc = nn.Linear(self.num_features, Config.NUM_CLASSES)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        return logits
