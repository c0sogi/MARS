import torch
import torch.nn as nn
from torchvision import models
from library.config import INPUT_CHANNELS, GROUPS, DROPOUT_RATE


class AsymmetricGroupedEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0.

    This model adapts the EfficientNet-B0 architecture for multi-modal MRI analysis.
    Key modifications:
    1. Grouped Convolutional Stem: Isolates the 4 modalities (FLAIR, T1w, T1wCE, T2w)
       in the first layer using 4 groups.
    2. Direct Asymmetric Initialization: Assigns pre-trained ImageNet filters directly
       to specific modality groups without interleaving.
    3. Volumetric Input: Accepts 12 channels (4 modalities x 3 slabs).
    """

    def __init__(self):
        super(AsymmetricGroupedEfficientNet, self).__init__()

        # Load pre-trained EfficientNet-B0 with ImageNet weights
        # Using string alias for compatibility with torchvision 0.13+
        self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")

        # Modify the stem to handle 12-channel input with grouped convolutions
        self._modify_stem()

        # Replace the classification head for binary prediction
        self._replace_head()

    def _modify_stem(self):
        """
        Surgically replaces the first convolutional layer.
        Preserves the subsequent BatchNorm and Activation layers.
        """
        # The stem in EfficientNet-B0 is the first block in 'features'
        # features[0] is a Conv2dNormActivation block
        # features[0][0] is the Conv2d layer
        original_conv = self.backbone.features[0][0]

        # Extract configuration from the original layer
        out_channels = original_conv.out_channels
        kernel_size = original_conv.kernel_size
        stride = original_conv.stride
        padding = original_conv.padding
        bias = original_conv.bias is not None

        # Create the new Grouped Convolution layer
        # in_channels=12, groups=4 -> 3 channels per group (matches original RGB 3 channels)
        new_conv = nn.Conv2d(
            in_channels=INPUT_CHANNELS,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=GROUPS,
            bias=bias,
        )

        # Direct Asymmetric Initialization (Direct Block Copy)
        # Original weights: (32, 3, 3, 3)
        # New weights: (32, 3, 3, 3) [since 12 input / 4 groups = 3]
        # We clone the weights directly. This assigns:
        # Filters 0-7  -> Group 0 (Modality 1)
        # Filters 8-15 -> Group 1 (Modality 2)
        # ... and so on.
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.clone()
            if bias:
                new_conv.bias.data = original_conv.bias.data.clone()

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

    def _replace_head(self):
        """
        Replaces the classifier with a regularized binary classification head.
        """
        # Access the input features of the final linear layer
        # classifier[1] is the Linear layer in the default configuration
        in_features = self.backbone.classifier[1].in_features

        # Define new head: Dropout -> Linear -> Logits
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_RATE, inplace=True),
            nn.Linear(in_features=in_features, out_features=1),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 12, 224, 224)
        Returns:
            torch.Tensor: Logits of shape (Batch, 1)
        """
        return self.backbone(x)
