import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import (
    TOTAL_INPUT_CHANNELS,
    NUM_GROUPS,
    DROP_RATE,
    BACKBONE_NAME,
    PRETRAINED,
)


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0.

    This model implements a Focal-Modality input strategy by modifying the stem
    of a pre-trained EfficientNet-B0. It accepts a 12-channel input tensor
    organized into 4 groups of 3 channels.

    Attributes:
        backbone (nn.Module): The modified EfficientNet-B0.
    """

    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        # 1. Load Pre-trained Backbone
        # Using IMAGENET1K_V1 weights as specified in the idea
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if PRETRAINED else None
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Modify Stem for Focal-Modality Input
        self._modify_stem()

        # 3. Reconstruct Classification Head
        # EfficientNet-B0 final feature map depth is 1280
        # The classifier block in torchvision implementation is a Sequential container
        # typically containing Dropout and Linear.
        original_classifier = self.backbone.classifier
        in_features = original_classifier[-1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=DROP_RATE, inplace=True),
            nn.Linear(in_features=in_features, out_features=1, bias=True),
        )

    def _modify_stem(self):
        """
        Replaces the first convolutional layer to support 12-channel input
        via Grouped Convolution, implementing Asymmetric Filter Distribution.
        """
        # Access the first block of the features Sequential container
        # features[0] is the Conv2dNormActivation block
        # features[0][0] is the Conv2d layer
        old_conv = self.backbone.features[0][0]

        # Create the new convolutional layer
        # in_channels: 12 (TOTAL_INPUT_CHANNELS)
        # out_channels: 32 (Preserved from original)
        # groups: 4 (NUM_GROUPS) -> Enforces 3 input channels per group
        new_conv = nn.Conv2d(
            in_channels=TOTAL_INPUT_CHANNELS,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            groups=NUM_GROUPS,
            bias=False,  # EfficientNet uses Batch Norm, so bias is False
        )

        # Asymmetric Filter Distribution / Weight Transfer
        # Original Weight Shape: (32, 3, 3, 3) -> (Out, In, K, K)
        # New Weight Shape:      (32, 3, 3, 3) -> (Out, In/Groups, K, K)
        # Since In/Groups = 12/4 = 3, the shapes are identical.
        # Direct assignment maps filters 0-7 to Group 1, 8-15 to Group 2, etc.
        if PRETRAINED:
            new_conv.weight.data = old_conv.weight.data.clone()

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 12, 224, 224).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        return self.backbone(x)
