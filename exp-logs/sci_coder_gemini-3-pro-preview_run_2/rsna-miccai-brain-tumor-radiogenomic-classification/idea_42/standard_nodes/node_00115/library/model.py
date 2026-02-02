import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class MILEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0.
    Cite solution_lesson_node_00114: Simplified from MIL to Single-Instance.

    Architecture:
    1. Backbone: EfficientNet-B0 initialized with ImageNet weights.
    2. Stem: Modified to accept 12 channels (4 modalities x 3 slices) using Grouped Convolutions (groups=4).
       Weights are initialized via Direct Block Copy from the original RGB weights.
    3. Head: Regularized with Dropout(p=0.5) and a single linear output.

    Input: (Batch, Channels, Height, Width)
    Output: (Batch, 1) - Logits
    """

    def __init__(self):
        super().__init__()

        # Load Pre-trained Backbone
        self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")

        # Modify the stem (first convolution)
        self._modify_stem()

        # Reconstruct the Classifier Head
        if (
            isinstance(self.backbone.classifier, nn.Sequential)
            and len(self.backbone.classifier) > 1
        ):
            in_features = self.backbone.classifier[1].in_features
        else:
            in_features = 1280

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(in_features, 1)
        )

    def _modify_stem(self):
        """
        Replaces the first convolutional layer with a Grouped Convolution.
        Performs Direct Asymmetric Initialization.
        """
        old_conv = self.backbone.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=Config.NUM_CHANNELS,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
            groups=Config.NUM_MODALITIES,
        )

        with torch.no_grad():
            new_conv.weight.data = old_conv.weight.data.clone()
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data.clone()

        self.backbone.features[0][0] = new_conv

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, H, W)

        Returns:
            torch.Tensor: Logits of shape (Batch, 1)
        """
        return self.backbone(x)
