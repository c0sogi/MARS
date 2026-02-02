import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class AsymmetricEfficientNet(nn.Module):
    """
    Implements the Asymmetric Grouped EfficientNet for Glioblastoma MGMT detection.

    This architecture utilizes an EfficientNet-B0 backbone with a modified stem
    to handle 4 MRI modalities (FLAIR, T1w, T1wCE, T2w) with 3 slices each (12 channels).
    It employs Grouped Convolutions to process modalities independently in the first layer
    and uses Direct Asymmetric Initialization to transfer pre-trained ImageNet features.
    """

    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        # 1. Load Pre-trained Backbone
        # We use EfficientNet-B0 pre-trained on ImageNet V1
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)

        # 2. Surgical Layer Replacement (Stem)
        # The stem in torchvision's EfficientNet is the first block in 'features'.
        # features[0] is a Conv2dNormActivation sequence: [Conv2d, BatchNorm2d, SiLU].
        # We target the Conv2d layer at index 0.
        old_conv = self.backbone.features[0][0]

        # Configuration for the new layer
        # Input: 12 Channels (4 modalities * 3 slices)
        # Groups: 4 (One group per modality to ensure isolation)
        in_channels = Config.TOTAL_CHANNELS
        out_channels = old_conv.out_channels
        groups = 4

        # Create the new Grouped Convolutional layer
        # We preserve the kernel size, stride, and padding to maintain geometry.
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
            groups=groups,
        )

        # Direct Asymmetric Initialization
        # Original weights shape: (Out=32, In/Groups=3, K=3, K=3) -> (32, 3, 3, 3)
        # New weights shape:      (Out=32, In/Groups=3, K=3, K=3) -> (32, 3, 3, 3)
        #
        # Since the shapes are identical, we perform a direct block copy.
        # This maps the 32 pre-trained ImageNet filters to the 4 modality groups sequentially:
        # - Filters 0-7  -> Group 0 (Modality 1: FLAIR)
        # - Filters 8-15 -> Group 1 (Modality 2: T1w)
        # - etc.
        # This avoids interleaving, preserving the semantic integrity of the filters.
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight)
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

        # 3. Regularized Head
        # We replace the default classifier to introduce a specific Dropout rate
        # and set the output dimension to 1 for binary classification.

        # Retrieve the input features of the final linear layer (1280 for EfficientNet-B0)
        last_linear = self.backbone.classifier[-1]
        in_features = last_linear.in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE, inplace=True), nn.Linear(in_features, 1)
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 12, H, W).

        Returns:
            torch.Tensor: Raw logits of shape (Batch, 1).
        """
        return self.backbone(x)
