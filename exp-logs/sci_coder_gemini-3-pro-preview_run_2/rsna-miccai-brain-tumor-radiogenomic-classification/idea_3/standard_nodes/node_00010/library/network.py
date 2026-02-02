import torch
import torch.nn as nn
import timm
from library import config


class EfficientNet25D(nn.Module):
    """
    2.5D EfficientNet for Volumetric MRI Classification.

    Architecture:
    1. Backbone: EfficientNet-B0
    2. Input Stem: Modified to accept 12 channels (3 slices * 4 modalities)
       using Grouped Convolutions (groups=4) to learn modality-specific features.
       Cite solution_lesson_node_00007.
    3. Head: Standard binary classification head.

    Design Choice:
    Reverted from Siamese architecture to single-branch ROI-focused model
    to reduce noise and overfitting. Cite solution_lesson_node_00009.
    """

    def __init__(self, backbone_name=config.BACKBONE, pretrained=True):
        super(EfficientNet25D, self).__init__()

        # 1. Load Backbone with Classifier
        # Initialize with in_chans=3 to load standard pre-trained RGB weights.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=config.NUM_CLASSES,
            in_chans=3,
        )

        # 2. Modify Input Stem for 12 Channels & Grouped Convolutions
        original_stem = self.backbone.conv_stem

        # Capture original configuration
        out_channels = original_stem.out_channels
        kernel_size = original_stem.kernel_size
        stride = original_stem.stride
        padding = original_stem.padding
        bias = original_stem.bias is not None

        # Create new stem
        # in_channels = 12 (config.IN_CHANNELS)
        # groups = 4 (One group per modality: FLAIR, T1w, T1wCE, T2w)
        # Weight shape for groups=4: (out_channels, in_channels/groups, k, k) -> (32, 3, 3, 3)
        self.new_stem = nn.Conv2d(
            in_channels=config.IN_CHANNELS,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=4,
            bias=bias,
        )

        # Initialize weights (Robust Initialization)
        # Cite solution_lesson_node_00006
        with torch.no_grad():
            self.new_stem.weight.copy_(original_stem.weight)
            if bias:
                self.new_stem.bias.copy_(original_stem.bias)

        # Replace the stem in the backbone
        self.backbone.conv_stem = self.new_stem

    def forward(self, x):
        """
        Forward Pass.

        Args:
            x (torch.Tensor): Content-Adaptive (ROI) View [B, 12, H, W]

        Returns:
            logits (torch.Tensor): Raw output scores [B, 1]
        """
        return self.backbone(x)
