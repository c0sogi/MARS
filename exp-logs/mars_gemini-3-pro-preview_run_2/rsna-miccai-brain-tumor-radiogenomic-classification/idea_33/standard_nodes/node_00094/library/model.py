import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class AsymmetricGroupedEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0.

    Implements:
    1. Surgical Stem Replacement: Replaces the first Conv2d layer to accept 12 channels
       (4 modalities x 3 slices) using Grouped Convolutions (groups=4).
    2. Asymmetric Filter Initialization: Distributes the 32 pre-trained ImageNet filters
       across the 4 modality groups to preserve feature diversity.
    3. Regularized Head: Replaces the classifier with Dropout(p=0.5) and a single output unit.
    """

    def __init__(self):
        super(AsymmetricGroupedEfficientNet, self).__init__()

        # 1. Load Pre-trained Backbone
        # We use IMAGENET1K_V1 weights for transfer learning
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)

        # 2. Surgical Stem Replacement
        # The first layer in EfficientNet-B0 is located at self.backbone.features[0][0]
        original_conv = self.backbone.features[0][0]

        # Extract original parameters
        out_channels = original_conv.out_channels
        kernel_size = original_conv.kernel_size
        stride = original_conv.stride
        padding = original_conv.padding
        bias = original_conv.bias

        # Configuration for the new stem
        # Input: 12 channels, Groups: 4
        # This results in 3 input channels per group, matching the original RGB kernel depth.
        in_channels = Config.INPUT_CHANNELS
        groups = Config.STEM_GROUPS

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias is not None,
        )

        # 3. Asymmetric Filter Initialization
        # Original weights shape: [32, 3, 3, 3] (Out, In/Groups, K, K)
        # New weights shape:      [32, 3, 3, 3] (Out, In/Groups, K, K) since 12/4 = 3.
        #
        # By directly copying the weights, we assign:
        # - Filters 0-7  to Group 0 (FLAIR)
        # - Filters 8-15 to Group 1 (T1w)
        # - etc.
        # This preserves the specific edge/texture detectors learned on ImageNet
        # but applies them to specific MRI modalities.
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.clone()
            if bias is not None:
                new_conv.bias.data = original_conv.bias.data.clone()

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

        # 4. Regularized Head Modification
        # The classifier in torchvision's EfficientNet is a Sequential block.
        # The last layer is the Linear projection.
        # We replace the entire classifier block to strictly control Dropout.

        # Get input features for the final linear layer (1280 for B0)
        num_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE, inplace=True),
            nn.Linear(num_features, Config.NUM_CLASSES),
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 12, 224, 224).
                              Channels are ordered: [FLAIR_0..2, T1w_0..2, T1wCE_0..2, T2w_0..2]

        Returns:
            torch.Tensor: Raw logits of shape (Batch, 1).
        """
        return self.backbone(x)
