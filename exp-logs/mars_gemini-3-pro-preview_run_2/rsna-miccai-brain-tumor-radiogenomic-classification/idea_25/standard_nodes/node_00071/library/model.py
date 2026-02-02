import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library import config


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0 with Dual-Stride 2.5D Fusion.

    This model adapts EfficientNet-B0 to accept a 24-channel volumetric input
    (4 modalities x 3 slices x 2 strides). It uses a grouped convolution stem
    to process 'Local' and 'Context' strides independently while sharing
    pre-trained feature detectors via asymmetric weight initialization.
    """

    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        # 1. Load Pre-trained Backbone
        # We use ImageNet V1 weights as the foundation for transfer learning
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Extract Original Stem Weights
        # The original stem is the first layer in the features Sequential block.
        # Structure: Conv2dNormActivation -> [Conv2d, BatchNorm, SiLU]
        original_conv = self.backbone.features[0][0]
        original_weights = original_conv.weight.data.clone()  # Shape: [32, 3, 3, 3]

        # 3. Construct Dual-Stage Stem
        # Layer 1: Grouped Convolution (Feature Extraction)
        # Input: 24 channels. Groups: 8.
        # Each group processes 3 channels (1 modality stack).
        # Output: 64 channels (8 groups * 8 filters/group).
        self.stem_grouped = nn.Conv2d(
            in_channels=config.INPUT_CHANNELS,  # 24
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=8,
            bias=False,
        )

        # Layer 2: Pointwise Convolution (Projection)
        # Projects 64 expanded features back to 32 to match EfficientNet backbone.
        self.stem_pointwise = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # Normalization and Activation
        self.stem_bn = nn.BatchNorm2d(32)
        self.stem_act = nn.SiLU(inplace=True)

        # Assemble the new stem block
        new_stem = nn.Sequential(
            self.stem_grouped, self.stem_pointwise, self.stem_bn, self.stem_act
        )

        # Replace the original stem in the backbone
        self.backbone.features[0] = new_stem

        # 4. Initialize Weights
        self.init_asymmetric_weights(original_weights)

        # 5. Modify Classifier Head
        # Replace the original classifier with a regularized binary head
        # Original: Sequential(Dropout, Linear(1280, 1000))
        # New: Sequential(Dropout, Linear(1280, 1))
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), nn.Linear(in_features, config.NUM_CLASSES)
        )

    def init_asymmetric_weights(self, original_weights):
        """
        Initializes the grouped convolution with pre-trained ImageNet weights.

        The 32 original filters are distributed across the first 4 groups (Local views)
        and then reused/replicated for the next 4 groups (Context views).
        """
        with torch.no_grad():
            # Target weight shape: [64, 3, 3, 3]
            # We have 32 original filters.

            # Assign to first 32 output channels (Groups 0-3: Local Stride)
            self.stem_grouped.weight.data[:32] = original_weights

            # Assign to next 32 output channels (Groups 4-7: Context Stride)
            self.stem_grouped.weight.data[32:] = original_weights

            # Initialize the pointwise projection layer
            # Using Kaiming Normal as it's a fresh projection layer
            nn.init.kaiming_normal_(
                self.stem_pointwise.weight, mode="fan_out", nonlinearity="relu"
            )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 24, 224, 224]

        Returns:
            torch.Tensor: Logits of shape [B, 1]
        """
        return self.backbone(x)
