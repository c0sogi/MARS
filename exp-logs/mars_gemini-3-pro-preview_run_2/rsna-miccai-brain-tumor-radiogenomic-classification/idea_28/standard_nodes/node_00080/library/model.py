import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0 for multi-modal MRI classification.

    Features:
    1. Grouped Convolutional Stem: Processes 4 modalities independently in the first layer.
    2. Asymmetric Filter Initialization: Distributes pre-trained ImageNet filters across groups.
    3. Regularized Head: Custom classifier with Dropout.
    """

    def __init__(self):
        super().__init__()

        # 1. Load Pre-trained Backbone
        # Use IMAGENET1K_V1 weights for robust feature extraction
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.base_model = efficientnet_b0(weights=weights)

        # 2. Surgical Layer Replacement (Stem)
        # The stem is the first layer in the 'features' sequential container.
        # Structure: features[0] -> Conv2dNormActivation -> [Conv2d, BatchNorm, Activation]
        original_conv_block = self.base_model.features[0]
        original_conv = original_conv_block[0]

        # Capture original weights (Shape: [32, 3, 3, 3])
        original_weights = original_conv.weight.data.clone()

        # Define new stem parameters based on Config and original layer
        in_channels = Config.IN_CHANNELS  # 12 (4 modalities * 3 slices)
        out_channels = original_conv.out_channels  # 32
        kernel_size = original_conv.kernel_size  # (3, 3)
        stride = original_conv.stride  # (2, 2)
        padding = original_conv.padding  # (1, 1)
        groups = Config.GROUPS  # 4 (One group per modality)
        bias = original_conv.bias is not None  # False

        # Create the new Grouped Convolutional Layer
        # Groups=4 means input is split into 4 chunks of 3 channels.
        # Each output filter only connects to one chunk.
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )

        # 3. Asymmetric Filter Initialization
        # We distribute the full bank of 32 pre-trained filters across the 4 modality groups.
        #
        # Logic:
        # - Original weights shape: [32, 3, 3, 3] (Out, In/Groups, K, K) where Groups=1
        # - New weights shape:      [32, 3, 3, 3] (Out, In/Groups, K, K) where Groups=4
        #   (New In = 12, so In/Groups = 12/4 = 3)
        #
        # Since the shapes are identical, we can directly copy the weights.
        # This effectively assigns:
        # - Filters 0-7  to Group 0 (FLAIR)
        # - Filters 8-15 to Group 1 (T1w)
        # - Filters 16-23 to Group 2 (T1wCE)
        # - Filters 24-31 to Group 3 (T2w)
        # This preserves the diversity of the original ImageNet detectors.

        new_conv.weight.data = original_weights

        # Replace the convolutional layer in the stem
        # We strictly preserve the subsequent BN and SiLU layers
        self.base_model.features[0][0] = new_conv

        # 4. Regularized Head
        # Reconstruct the classifier to include explicit Dropout
        # EfficientNet B0 final feature map depth is 1280
        last_channel = self.base_model.classifier[1].in_features

        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE, inplace=True),
            nn.Linear(last_channel, Config.NUM_CLASSES),
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 12, 224, 224)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.base_model(x)
