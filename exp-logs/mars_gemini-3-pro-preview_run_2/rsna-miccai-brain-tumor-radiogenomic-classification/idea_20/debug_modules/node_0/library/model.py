import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0.

    This model adapts EfficientNet-B0 for 12-channel input (4 modalities x 3 slices)
    using a Grouped Convolutional Stem to isolate modalities initially.

    It employs 'Asymmetric Filter Initialization' to distribute pre-trained ImageNet
    filters across the modality groups, preserving feature diversity.
    """

    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        # 1. Load Pretrained Backbone
        # We use IMAGENET1K_V1 weights as the baseline for transfer learning
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Capture Original Stem Weights
        # The first layer in EfficientNet-B0 (torchvision) is inside features[0][0]
        # features[0] is Conv2dNormActivation, index 0 is the Conv2d
        original_stem = self.backbone.features[0][0]
        original_weights = original_stem.weight.data.clone()

        # 3. Modify Stem for Grouped Convolutions
        # Original: In=3, Out=32, Groups=1
        # New:      In=12, Out=32, Groups=4
        # This enforces that each modality (3 channels) is convolved with its own set of filters
        out_channels = original_stem.out_channels
        kernel_size = original_stem.kernel_size
        stride = original_stem.stride
        padding = original_stem.padding

        # Create the new layer
        new_stem = nn.Conv2d(
            in_channels=Config.TOTAL_CHANNELS,  # 12
            out_channels=out_channels,  # 32
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=4,  # 4 Modalities
            bias=False,  # EfficientNet uses BN, so no bias in Conv
        )

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_stem

        # 4. Asymmetric Filter Initialization
        self.init_weights(original_weights)

        # 5. Reconstruct Classification Head
        # Replace the original classifier (Dropout+Linear) to ensure correct Dropout rate and Output size
        # Original B0 final linear layer input features is 1280
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE, inplace=True),
            nn.Linear(in_features, Config.NUM_CLASSES),
        )

    def init_weights(self, original_weights):
        """
        Implements Asymmetric Filter Initialization.

        The standard RGB stem has weights of shape (32, 3, 3, 3) -> (Out, In, K, K).
        The new grouped stem has weights of shape (32, 3, 3, 3) -> (Out, In/Groups, K, K).

        Since the shapes are identical, we can directly copy the weights.
        Semantically, this maps:
        - Filters 0-7   -> Group 1 (FLAIR)
        - Filters 8-15  -> Group 2 (T1w)
        - Filters 16-23 -> Group 3 (T1wCE)
        - Filters 24-31 -> Group 4 (T2w)

        This distributes the diverse feature detectors learned on ImageNet across
        the different MRI modalities.
        """
        with torch.no_grad():
            self.backbone.features[0][0].weight.data.copy_(original_weights)

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 12, 224, 224)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.backbone(x)
