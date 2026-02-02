import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library import config


class AsymmetricEfficientNet(nn.Module):
    """
    EfficientNet-B0 with an Asymmetric Grouped Convolution Stem.

    Architecture:
    - Backbone: EfficientNet-B0 (Pre-trained on ImageNet)
    - Stem: Modified to accept 12 channels (4 modalities x 3 slices) using Grouped Convolutions (groups=4).
    - Initialization: Direct Block Copy of ImageNet weights to preserve feature extraction capabilities.
    - Head: Regularized with Dropout(p=0.5) and a single linear output.
    """

    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        # 1. Load Pre-trained Backbone
        # We use the standard V1 weights as per general stability recommendations
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Adapt Stem for Multi-Modal Input
        self._adapt_stem()

        # 3. Modify Classifier Head
        # EfficientNet B0 classifier structure:
        # Sequential(Dropout(p=0.2), Linear(in_features=1280, out_features=1000))
        # We replace it with p=0.5 and out_features=1
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=config.DROPOUT_RATE, inplace=True),
            nn.Linear(in_features=in_features, out_features=1, bias=True),
        )

    def _adapt_stem(self):
        """
        Surgically replaces the first convolutional layer.
        - Original: Conv2d(3, 32, kernel=3, stride=2, padding=1, bias=False)
        - New:      Conv2d(12, 32, kernel=3, stride=2, padding=1, groups=4, bias=False)

        Performs Direct Block Copy initialization.
        """
        # In torchvision's efficientnet_b0, features[0] is Conv2dNormActivation
        # features[0][0] is the Conv2d layer
        old_conv = self.backbone.features[0][0]

        # Verify assumptions about the old layer
        if not isinstance(old_conv, nn.Conv2d):
            raise TypeError(f"Expected Conv2d at features[0][0], got {type(old_conv)}")

        # Define new configuration
        in_channels = config.NUM_CHANNELS  # 12
        out_channels = old_conv.out_channels  # 32
        kernel_size = old_conv.kernel_size  # (3, 3)
        stride = old_conv.stride  # (2, 2)
        padding = old_conv.padding  # (1, 1)
        groups = config.STEM_GROUPS  # 4
        bias = old_conv.bias is not None

        # Create new layer
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )

        # Direct Block Copy Initialization
        # Old weights shape: (32, 3, 3, 3) -> (Out, In, K, K) [Groups=1]
        # New weights shape: (32, 3, 3, 3) -> (Out, In/Groups, K, K) [Groups=4]
        # Since 12 input channels / 4 groups = 3 channels per group, the shapes match perfectly.
        # We copy the weights directly. This assigns:
        # - Filters 0-7 to Group 1 (Channels 0-2)
        # - Filters 8-15 to Group 2 (Channels 3-5)
        # - etc.
        with torch.no_grad():
            new_conv.weight.data = old_conv.weight.data.clone()
            if bias:
                new_conv.bias.data = old_conv.bias.data.clone()

        # Replace the layer
        self.backbone.features[0][0] = new_conv

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 12, 224, 224)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.backbone(x)
