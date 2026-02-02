import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0.

    Architectural Changes:
    1. Grouped Convolutional Stem (Groups=4) for modality isolation.
    2. Direct Block Copy initialization for stem weights.
    3. Regularized Classification Head (Dropout p=0.5).
    """

    def __init__(self):
        super().__init__()

        # Load pre-trained EfficientNet-B0
        # We use the default IMAGENET1K_V1 weights
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)

        # ----------------------------------------------------------------------
        # 1. Modify Stem: Grouped Convolution
        # ----------------------------------------------------------------------
        # The stem is located at self.backbone.features[0]
        # It is a Conv2dNormActivation containing: (0) Conv, (1) BN, (2) SiLU
        original_conv = self.backbone.features[0][0]

        # Extract parameters from original conv
        out_channels = original_conv.out_channels  # 32
        kernel_size = original_conv.kernel_size  # (3, 3)
        stride = original_conv.stride  # (2, 2)
        padding = original_conv.padding  # (1, 1)
        bias = original_conv.bias  # False

        # Configuration for Asymmetric Input
        # Input channels: 12 (4 modalities * 3 slices)
        # Groups: 4 (one group per modality)
        in_channels = Config.INPUT_CHANNELS
        groups = len(Config.MODALITIES)

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )

        # ----------------------------------------------------------------------
        # 2. Direct Block Copy Initialization
        # ----------------------------------------------------------------------
        # Copy weights from the original RGB stem to the new grouped stem.
        # Original weight shape: (32, 3, 3, 3) -> (Out, In/Groups, K, K) where Groups=1, In=3
        # New weight shape:      (32, 3, 3, 3) -> (Out, In/Groups, K, K) where Groups=4, In=12
        # The shapes align perfectly, allowing direct assignment.
        with torch.no_grad():
            new_conv.weight.copy_(original_conv.weight)

        # Replace the conv layer in the stem sequence
        self.backbone.features[0][0] = new_conv

        # ----------------------------------------------------------------------
        # 3. Regularized Head
        # ----------------------------------------------------------------------
        # Replace classifier: Dropout(0.5) -> Linear(1280, 1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True), nn.Linear(in_features, 1)
        )

    def forward(self, x):
        # Forward pass through the modified backbone
        # This executes features (Stem -> SE -> Blocks) -> AvgPool -> Flatten -> Classifier
        return self.backbone(x)
