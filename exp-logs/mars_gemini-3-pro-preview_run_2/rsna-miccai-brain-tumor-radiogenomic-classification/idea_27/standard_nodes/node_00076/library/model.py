import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0 with Cross-Modality Stacking.

    This model implements the 'Asymmetric Filter Distribution' strategy:
    - Input: 12 Channels (4 Groups x 3 Channels).
    - Stem: Grouped Convolution (groups=4) to process each stack independently
      at the pixel level while fusing them at the feature level in deeper layers.
    - Weights: Pre-trained ImageNet filters are distributed across the 4 groups
      to preserve the full bank of edge/texture detectors.
    """

    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        # 1. Load Pre-trained Backbone
        # We use the standard IMAGENET1K_V1 weights for robust feature extraction
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)

        # 2. Modify Stem for Asymmetric Grouped Input
        # Original Stem: Conv2d(3, 32, kernel=3, stride=2, padding=1, bias=False)
        original_stem = self.backbone.features[0][0]

        # New Stem: Conv2d(12, 32, ..., groups=4)
        # We use groups=4 to split the 12 input channels into 4 groups of 3.
        # This strictly adheres to the "Rule of 3" geometry of the pre-trained kernels.
        new_stem = nn.Conv2d(
            in_channels=Config.INPUT_CHANNELS,  # 12
            out_channels=original_stem.out_channels,  # 32
            kernel_size=original_stem.kernel_size,  # 3
            stride=original_stem.stride,  # 2
            padding=original_stem.padding,  # 1
            bias=original_stem.bias is not None,  # False
            groups=4,  # Key architectural change
        )

        # 3. Asymmetric Filter Distribution (Weight Transfer)
        # Original weights shape: [32, 3, 3, 3] (Out, In/Groups, K, K) where Groups=1
        # New weights shape:      [32, 3, 3, 3] (Out, In/Groups, K, K) where Groups=4 (12/4=3)
        #
        # In PyTorch, grouped convolution weights map output channels to input groups sequentially.
        # Filters 0-7   -> Input Group 1 (Channels 0-2:   Slice -5 [FLAIR, T1w, T1wCE])
        # Filters 8-15  -> Input Group 2 (Channels 3-5:   Slice  0 [FLAIR, T1w, T1wCE])
        # Filters 16-23 -> Input Group 3 (Channels 6-8:   Slice +5 [FLAIR, T1w, T1wCE])
        # Filters 24-31 -> Input Group 4 (Channels 9-11:  Context  [T2w, T2w, T2w])
        #
        # By cloning the weights, we effectively distribute the pre-trained detectors.
        with torch.no_grad():
            new_stem.weight.data = original_stem.weight.data.clone()

        # Replace the stem in the backbone
        self.backbone.features[0][0] = new_stem

        # 4. Modify Classifier Head
        # Original: Sequential(Dropout(0.2), Linear(1280, 1000))
        # New: Sequential(Dropout(Config.DROPOUT_RATE), Linear(1280, 1))

        # Retrieve the input features dimension of the final linear layer
        # In torchvision's EfficientNet, classifier is a Sequential block where index 1 is the Linear layer.
        final_in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE, inplace=True),
            nn.Linear(final_in_features, 1),
        )

    def forward(self, x):
        # Forward pass through the modified backbone
        # Output is a single logit (unscaled) per sample
        return self.backbone(x)
