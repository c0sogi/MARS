import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import library.config as C


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0 with Multi-Modal Input.

    Architecture:
    - Backbone: EfficientNet-B0
    - Input: 12 Channels (4 groups of 3: FLAIR, T1w, T1wCE, T2w)
    - Stem: Grouped Convolution (groups=4) enforcing modality isolation.
    - Initialization: Direct Asymmetric (Block Copy of ImageNet weights).
    - Head: Dropout(0.5) -> Linear(1).
    """

    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        # 1. Load Pretrained Backbone
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if C.PRETRAINED else None
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Modify Stem for 12-channel Grouped Input
        self._modify_stem()

        # 3. Modify Classification Head
        self._modify_head()

    def _modify_stem(self):
        """
        Replaces the first convolutional layer to handle 12-channel input
        using grouped convolutions for modality isolation.
        Performs Direct Asymmetric Initialization.
        """
        # Access the original first layer
        # EfficientNet-B0 structure: features -> [0] (ConvNormActivation) -> [0] (Conv2d)
        original_conv = self.backbone.features[0][0]

        # Verify assumptions about original layer
        out_channels = original_conv.out_channels  # Should be 32
        kernel_size = original_conv.kernel_size  # Should be (3, 3)
        stride = original_conv.stride  # Should be (2, 2)
        padding = original_conv.padding  # Should be (1, 1)

        # Create new grouped convolution
        # Input: 12 channels
        # Groups: 4 (So each group sees 12/4 = 3 channels)
        # This matches the kernel depth of the original ImageNet weights (3 channels).
        new_conv = nn.Conv2d(
            in_channels=C.INPUT_CHANNELS,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=4,  # Enforce modality isolation
            bias=False,
        )

        # Direct Asymmetric Initialization
        # Original weights shape: [32, 3, 3, 3] (Out, In, K, K)
        # New weights shape:      [32, 3, 3, 3] (Out, In/Groups, K, K)
        #
        # By directly copying, we assign:
        # Filters 0-7   -> Group 1 (FLAIR)
        # Filters 8-15  -> Group 2 (T1w)
        # Filters 16-23 -> Group 3 (T1wCE)
        # Filters 24-31 -> Group 4 (T2w)
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.clone()

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

    def _modify_head(self):
        """
        Replaces the classifier head with Dropout and a Linear projection.
        """
        # EfficientNet classifier is a Sequential block.
        # Index 1 is the Linear layer.
        original_classifier = self.backbone.classifier
        in_features = original_classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=C.DROPOUT_RATE, inplace=True),
            nn.Linear(in_features, C.NUM_CLASSES),
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
