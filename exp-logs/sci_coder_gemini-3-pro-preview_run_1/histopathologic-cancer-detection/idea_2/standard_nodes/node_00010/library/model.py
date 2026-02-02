import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet121_Weights
from library.config import Config


class ModifiedDenseNet(nn.Module):
    """
    A DenseNet121-based architecture modified for small input resolutions (e.g., 48x48).

    Modifications:
    1. Input Stem: Replaces the standard 7x7 stride-2 conv and 3x3 stride-2 maxpool
       with a 3x3 stride-1 conv and Identity pooling. This preserves spatial resolution.
    2. Classifier: Replaces the ImageNet classifier with a binary classification head.
    """

    def __init__(self, pretrained=True):
        """
        Args:
            pretrained (bool): If True, loads ImageNet weights for the backbone.
        """
        super(ModifiedDenseNet, self).__init__()

        # Load the base DenseNet121 model
        if pretrained:
            weights = DenseNet121_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.model = models.densenet121(weights=weights)

        # =====================================================================
        # Modification 1: Input Stem
        # =====================================================================
        # The standard DenseNet starts with:
        #   Conv2d(3, 64, kernel=7, stride=2, padding=3) -> 2x downsample
        #   MaxPool2d(kernel=3, stride=2, padding=1)     -> 2x downsample
        # Total downsample: 4x. For 48x48 input, this results in 12x12 features
        # entering the first block, which is too small.

        # Replace Conv0 with 3x3 stride-1
        self.model.features.conv0 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Replace Pool0 with Identity (no downsampling)
        self.model.features.pool0 = nn.Identity()

        # =====================================================================
        # Modification 2: Classifier
        # =====================================================================
        # Replace the 1000-class linear layer with a binary output (1 class)
        # Input features for DenseNet121 classifier is 1024
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, 1)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)

        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.model(x)
