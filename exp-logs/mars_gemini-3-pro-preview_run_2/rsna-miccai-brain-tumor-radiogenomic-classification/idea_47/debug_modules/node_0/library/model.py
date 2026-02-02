import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation Block for channel-wise feature recalibration.
    Mechanism: Global Average Pooling -> Reduction FC -> Expansion FC -> Sigmoid -> Channel Scaling.
    """

    def __init__(self, input_channels, squeeze_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.SiLU(inplace=True)  # EfficientNet uses SiLU (Swish)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0 with Stem-Level Channel Attention.

    Architectural Changes:
    1. Grouped Convolutional Stem (Groups=4) for modality isolation.
    2. Direct Block Copy initialization for stem weights.
    3. Stem-Level SE Block inserted after the stem.
    4. Regularized Classification Head (Dropout p=0.5).
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
        # 3. Insert Stem-Level SE Block
        # ----------------------------------------------------------------------
        # Insert SE block after the stem (Conv+BN+SiLU) and before the first MBConv.
        # Stem output channels = 32.
        # Reduction ratio ~4 -> squeeze channels = 8.
        se_block = SqueezeExcitation(input_channels=32, squeeze_channels=8)

        # Reconstruct the features Sequential container
        # features[0] is the Stem
        # features[1:] are the MBConv blocks
        layers = [self.backbone.features[0], se_block]
        layers.extend(
            [self.backbone.features[i] for i in range(1, len(self.backbone.features))]
        )

        self.backbone.features = nn.Sequential(*layers)

        # ----------------------------------------------------------------------
        # 4. Regularized Head
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
