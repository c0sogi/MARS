import torch
import torch.nn as nn
from torchvision import models
from library import config


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0.

    This model adapts a pre-trained EfficientNet-B0 to accept 12-channel inputs
    (4 modalities x 3 slices) using Grouped Convolutions.

    Key Innovation:
    - Asymmetric Filter Distribution: Instead of repeating the same weights,
      the original 32 ImageNet filters are distributed across the 4 modality groups.
      - Filters 0-7   -> FLAIR
      - Filters 8-15  -> T1w
      - Filters 16-23 -> T1wCE
      - Filters 24-31 -> T2w
    """

    def __init__(self, num_classes=config.NUM_CLASSES, in_channels=config.IN_CHANNELS):
        super(AsymmetricEfficientNet, self).__init__()

        # 1. Load Pre-trained Backbone
        # We use IMAGENET1K_V1 weights as the foundation
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)

        # 2. Modify First Convolutional Layer (Stem)
        # Original: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        original_conv = self.backbone.features[0][0]

        # New: Conv2d(12, 32, groups=4, ...)
        # We use groups=4 to enforce independent processing of the 4 modalities
        # in the first layer.
        # in_channels=12, out_channels=32, groups=4
        # This implies 12/4 = 3 input channels per group (matching RGB depth).
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
            groups=4,  # 4 Modalities
        )

        # 3. Asymmetric Filter Distribution
        self._init_asymmetric_weights(original_conv, new_conv)

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

        # 4. Modify Classification Head
        # EfficientNet classifier structure: Sequential(Dropout, Linear)
        # We retain the Dropout layer as requested for regularization.
        original_dropout = self.backbone.classifier[0]
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            original_dropout, nn.Linear(in_features, num_classes)
        )

    def _init_asymmetric_weights(self, original_conv, new_conv):
        """
        Distributes the 32 pre-trained filters across the 4 groups.

        PyTorch Grouped Conv2d weights shape: (Out, In/Groups, K, K)
        Original shape: (32, 3, 3, 3)
        New shape:      (32, 3, 3, 3)  [since 12 input / 4 groups = 3]

        With groups=4, output channels are assigned sequentially to groups:
        - Out 0-7   connect to In 0-2 (Group 0)
        - Out 8-15  connect to In 3-5 (Group 1)
        - Out 16-23 connect to In 6-8 (Group 2)
        - Out 24-31 connect to In 9-11 (Group 3)

        By copying the weights directly, we assign:
        - Original filters 0-7  -> Group 0 (FLAIR)
        - Original filters 8-15 -> Group 1 (T1w)
        - etc.
        """
        with torch.no_grad():
            # Direct copy achieves the specific distribution strategy described
            new_conv.weight.data = original_conv.weight.data.clone()

            # Handle bias if it exists (EfficientNet convs usually don't have bias due to BN)
            if original_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data.clone()

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 12, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.backbone(x)
