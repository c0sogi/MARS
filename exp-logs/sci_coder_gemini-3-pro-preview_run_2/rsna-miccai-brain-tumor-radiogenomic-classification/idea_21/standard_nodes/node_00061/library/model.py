import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0.

    This model modifies the standard EfficientNet-B0 stem to accept 12-channel input
    (4 modalities * 3 slices) using Grouped Convolutions.

    Key Feature: Asymmetric Filter Initialization
    Instead of averaging weights, we distribute the 32 pre-trained ImageNet filters
    across the 4 modality groups. Since the weight shapes match exactly between
    standard Conv2d(3, 32) and Grouped Conv2d(12, 32, groups=4), we can preserve
    the full diversity of pre-trained feature detectors.
    """

    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        # 1. Load Pre-trained Backbone
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if Config.PRETRAINED else None
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Modify Stem (First Convolutional Layer)
        # Access the first layer of the features block
        original_stem = self.backbone.features[0][0]

        # Configuration check
        in_channels = Config.IN_CHANNELS  # 12
        out_channels = original_stem.out_channels  # 32
        groups = Config.STEM_GROUPS  # 4

        # Create the new Grouped Convolution stem
        # Note: EfficientNet uses bias=False because it's followed by BatchNorm
        new_stem = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            groups=groups,
            bias=False,
        )

        # 3. Asymmetric Filter Initialization
        if Config.PRETRAINED and Config.ASYMMETRIC_INIT:
            # Original weights shape: (32, 3, 3, 3) -> (Out, In, K, K)
            # New weights shape: (32, 12/4, 3, 3) -> (32, 3, 3, 3) -> (Out, In/Groups, K, K)
            # The shapes are identical. We copy the weights directly.
            # This assigns filters 0-7 to Modality 0 (FLAIR), 8-15 to Modality 1 (T1w), etc.
            with torch.no_grad():
                new_stem.weight.data = original_stem.weight.data.clone()

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_stem

        # 4. Modify Classification Head
        # EfficientNet's classifier block is usually:
        # (0): Dropout
        # (1): Linear
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE, inplace=True),
            nn.Linear(in_features, Config.NUM_CLASSES),
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 12, 224, 224)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.backbone(x)
