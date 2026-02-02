import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class ProjectionStem(nn.Module):
    """
    A stabilized projection block that compresses high-density volumetric input
    (128 channels) into a feature space compatible with the backbone (64 channels).

    Structure: Conv2d (128->64) -> BatchNorm -> ReLU
    Initialization: Kaiming Normal (He Init)
    """

    def __init__(self, in_channels, out_channels):
        super(ProjectionStem, self).__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Explicit Kaiming Initialization for stability
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Stabilized25DNet(nn.Module):
    """
    Stabilized High-Density 2.5D Network.

    1. Input: (B, 128, 256, 256) - 32 slices * 4 modalities.
    2. Stem: Projects 128 channels -> 64 channels.
    3. Backbone: EfficientNet-B0 (pretrained), modified to accept 64 channels.
    4. Head: Linear layer for binary classification (logits).
    """

    def __init__(self):
        super(Stabilized25DNet, self).__init__()

        # Dimensions from Config
        # 32 slices * 4 modalities = 128 channels
        input_channels = Config.IN_CHANNELS
        stem_channels = 64

        # 1. Stabilized Projection Stem
        self.stem = ProjectionStem(input_channels, stem_channels)

        # 2. Backbone: EfficientNet-B0
        # Load with default weights (ImageNet)
        self.backbone = models.efficientnet_b0(weights="DEFAULT")

        # Modify the first convolutional layer to accept stem_channels (64) instead of 3
        # In torchvision's EfficientNet, features[0] is a Conv2dNormActivation block
        # features[0][0] is the Conv2d layer.
        original_first_conv = self.backbone.features[0][0]

        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=stem_channels,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=original_first_conv.bias,
        )

        # Initialize the new layer using Kaiming Normal to match the stem's stability
        nn.init.kaiming_normal_(
            self.backbone.features[0][0].weight, mode="fan_out", nonlinearity="relu"
        )

        # 3. Classification Head
        # Replace the default classifier (Dropout + Linear(1280 -> 1000))
        # with Dropout + Linear(1280 -> 1)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), nn.Linear(num_features, 1)
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 128, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        # 1. Pass through Stem
        x = self.stem(x)

        # 2. Pass through Backbone (Features + Pool + Classifier)
        # torchvision EfficientNet forward implementation handles pooling and flattening internally
        logits = self.backbone(x)

        return logits
