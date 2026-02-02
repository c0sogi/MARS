import torch
import torch.nn as nn
import timm
from library.config import Config


class DenseNet121Custom(nn.Module):
    """
    DenseNet121 architecture adapted for small resolution histopathology patches.

    Cite solution_lesson_node_00008: Partial Transfer Learning with Stem Modification.
    Cite solution_lesson_node_00003: Adapt Standard Backbone Stems for Small Input Resolutions.
    """

    def __init__(self, pretrained: bool = Config.PRETRAINED):
        super().__init__()

        # Load DenseNet121 backbone
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=pretrained, num_classes=0
        )

        # ---------------------------------------------------------------------
        # Stem Modification
        # ---------------------------------------------------------------------
        # Original Stem:
        # conv0: 7x7, stride 2, padding 3
        # norm0: BatchNorm
        # relu0: ReLU
        # pool0: 3x3 maxpool, stride 2

        # We modify conv0 to 3x3 stride 1 to preserve resolution.
        # We replace pool0 with Identity to prevent early downsampling.

        # Access features sequential container
        # Note: timm's DenseNet implementation exposes 'features'

        # 1. Modify Conv0
        original_conv = self.backbone.features.conv0
        new_conv = nn.Conv2d(
            in_channels=original_conv.in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=original_conv.bias is not None,
        )
        self.backbone.features.conv0 = new_conv

        # 2. Remove Pool0
        self.backbone.features.pool0 = nn.Identity()

        # ---------------------------------------------------------------------
        # Classifier Head
        # ---------------------------------------------------------------------
        self.num_features = self.backbone.num_features
        self.fc = nn.Linear(self.num_features, Config.NUM_CLASSES)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        return logits
