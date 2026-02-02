import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class GroupedEfficientNet(nn.Module):
    """
    A customized EfficientNet-B0 for multi-modal MRI analysis.

    Key Modifications:
    1. Grouped Stem: The first convolutional layer is replaced to handle 12 input channels
       (4 modalities * 3 slices) using Grouped Convolutions (groups=4). This forces
       independent feature extraction for each modality in the initial stage (Cite Lesson 00007).
    2. Robust Initialization: The grouped stem is initialized with pre-trained RGB weights
       replicated across the groups to preserve texture/edge detection capabilities (Cite Lesson 00006).
    """

    def __init__(self):
        super(GroupedEfficientNet, self).__init__()

        # Load pre-trained EfficientNet-B0
        # 'DEFAULT' loads the best available weights (IMAGENET1K_V1 or similar)
        self.backbone = models.efficientnet_b0(weights="DEFAULT")

        # ----------------------------------------------------------------------
        # 1. Modality-Grouped Stem Modification
        # ----------------------------------------------------------------------
        # Access the original first layer
        # Structure: features[0][0] is the Conv2d, features[0][1] is BN, features[0][2] is SiLU
        original_stem = self.backbone.features[0][0]

        # Create new stem
        # Input: 12 channels (Config.TOTAL_CHANNELS)
        # Groups: 4 (One per modality)
        # Output: 32 channels (Standard for B0)
        # Kernel/Stride/Padding: Same as original
        new_stem = nn.Conv2d(
            in_channels=Config.TOTAL_CHANNELS,
            out_channels=original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            bias=original_stem.bias is not None,
            groups=4,
        )

        # ----------------------------------------------------------------------
        # 2. Robust Weight Initialization
        # ----------------------------------------------------------------------
        # Original weight shape: (32, 3, 3, 3) -> [Out, In, K, K]
        # New weight shape with groups=4: (32, 12/4, 3, 3) -> (32, 3, 3, 3)
        #
        # Since the shapes are identical, we can directly copy the pre-trained weights.
        # This effectively "replicates" the RGB filters for each of the 4 modality groups,
        # ensuring that T1, T2, FLAIR, etc., all start with high-quality feature detectors.
        with torch.no_grad():
            new_stem.weight.copy_(original_stem.weight)

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_stem

        # ----------------------------------------------------------------------
        # 3. Classifier Head Modification
        # ----------------------------------------------------------------------
        # The classifier in EfficientNet is a Sequential block.
        # Index [1] is the final Linear layer.
        original_fc = self.backbone.classifier[1]

        # Create new linear layer for binary classification (output size 1)
        self.backbone.classifier[1] = nn.Linear(
            in_features=original_fc.in_features, out_features=Config.NUM_CLASSES
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 12, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        return self.backbone(x)
