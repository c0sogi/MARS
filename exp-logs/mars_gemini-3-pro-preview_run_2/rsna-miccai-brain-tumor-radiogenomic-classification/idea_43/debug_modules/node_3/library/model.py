import torch
import torch.nn as nn
import timm
from library.config import Config


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0 with Modality-Adaptive Stride support.

    Implements the architectural changes described in Idea 43:
    1. Input: 12 channels (4 groups of 3).
    2. Stem: Grouped Convolution (groups=4) to enforce modality isolation.
    3. Initialization: Direct Block Copy of ImageNet weights (Non-interleaved).
    4. Head: Regularized with Dropout (p=0.5).
    """

    def __init__(self, pretrained=True):
        super(AsymmetricEfficientNet, self).__init__()

        # Load EfficientNet-B0 backbone using timm
        # We initialize with num_classes=1 to get the correct feature dimension,
        # though we will rebuild the head to include dropout.
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=1
        )

        # ----------------------------------------------------------------------
        # 1. Surgical Stem Replacement (Grouped Convolution)
        # ----------------------------------------------------------------------
        # Retrieve the original stem convolution
        old_stem = self.backbone.conv_stem

        # Define the new stem:
        # - in_channels=12 (Config.NUM_CHANNELS)
        # - groups=4 (One group per modality: FLAIR, T2w, T1w, T1wCE)
        # - Preserves kernel_size, stride, padding from the original B0 stem.
        new_stem = nn.Conv2d(
            in_channels=Config.NUM_CHANNELS,
            out_channels=old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            groups=4,
            bias=(old_stem.bias is not None),
        )

        # ----------------------------------------------------------------------
        # 2. Direct Block Copy Initialization
        # ----------------------------------------------------------------------
        # We assign pre-trained ImageNet filters to the groups sequentially.
        # Standard Stem Weights: (32, 3, 3, 3)
        # New Stem Weights:      (32, 12//4, 3, 3) -> (32, 3, 3, 3)
        #
        # Since the shapes match, a direct clone performs the assignment:
        # Filters 0-7   -> Group 1 (FLAIR)
        # Filters 8-15  -> Group 2 (T2w)
        # Filters 16-23 -> Group 3 (T1w)
        # Filters 24-31 -> Group 4 (T1wCE)
        if pretrained:
            new_stem.weight.data = old_stem.weight.data.clone()
            if old_stem.bias is not None:
                new_stem.bias.data = old_stem.bias.data.clone()

        # Replace the stem in the backbone
        self.backbone.conv_stem = new_stem

        # ----------------------------------------------------------------------
        # 3. Regularized Head
        # ----------------------------------------------------------------------
        # Reconstruct the classification head to include Dropout (p=0.5)
        # before the final linear projection.
        original_classifier = self.backbone.classifier
        in_features = original_classifier.in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(in_features, 1)
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 12, 224, 224).
        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        return self.backbone(x)
