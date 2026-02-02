import torch
import torch.nn as nn
from torchvision import models


class FactorizedStem(nn.Module):
    """
    Factorized Projection Stem for efficient dimensionality reduction of high-channel inputs.
    Consists of a Depthwise Separable Convolution block:
    - Depthwise Conv: Learns spatial features per slice/modality.
    - Pointwise Conv: Performs early fusion across channels.
    """

    def __init__(self, in_channels=128, out_channels=64):
        super().__init__()
        # Depthwise Convolution: (B, 128, H, W) -> (B, 128, H, W)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Pointwise Convolution: (B, 128, H, W) -> (B, 64, H, W)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        # Kaiming/He Normal initialization for stability
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class MGMTNet(nn.Module):
    """
    Factorized Stabilized High-Density 2.5D Network.
    Integrates a Factorized Projection Stem with an EfficientNet-B0 backbone.
    """

    def __init__(self):
        super().__init__()

        # 1. Factorized Projection Stem
        # Compresses 128 input channels (32 slices * 4 modalities) to 64
        self.stem = FactorizedStem(in_channels=128, out_channels=64)

        # 2. Backbone: EfficientNet-B0
        # Initialize with ImageNet weights
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)

        # Modify the first layer to accept 64 channels from the stem
        # Original first layer: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        original_first_conv = self.backbone.features[0][0]

        new_first_conv = nn.Conv2d(
            in_channels=64,  # Output of stem
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=original_first_conv.bias is not None,
        )

        # Initialize the new first layer
        nn.init.kaiming_normal_(
            new_first_conv.weight, mode="fan_out", nonlinearity="relu"
        )

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_first_conv

        # 3. Head
        # Replace the default classifier (Dropout + Linear) with a single Linear layer.
        # Global Average Pooling is handled by self.backbone.avgpool before the classifier.
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Linear(num_features, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 128, 256, 256)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        # Pass through Factorized Stem
        x = self.stem(x)

        # Pass through Backbone
        # EfficientNet forward: features -> avgpool -> flatten -> classifier
        logits = self.backbone(x)

        return logits
