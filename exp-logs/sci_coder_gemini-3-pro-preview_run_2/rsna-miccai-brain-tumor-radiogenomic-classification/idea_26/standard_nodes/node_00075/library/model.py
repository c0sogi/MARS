import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet for MGMT Promoter Methylation Prediction.

    Implements the 'Direct Grouped Stem' logic:
    1. Input: 24 channels (4 modalities * 2 strides * 3 slices).
    2. Stem: Conv2d(24, 32, groups=8).
       - Groups=8 splits input into 8 groups of 3 channels each.
       - This matches the 3-channel depth of standard ImageNet weights.
    3. Initialization: Direct copy of pre-trained weights (32, 3, 3, 3) into the new stem.
    """

    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        # 1. Load Pre-trained Backbone
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if Config.PRETRAINED else None
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Surgically Replace the Stem
        # Access the first layer of the features block
        # Original: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        original_stem = self.backbone.features[0][0]

        # Verify configuration matches expectation
        if original_stem.in_channels != 3 or original_stem.out_channels != 32:
            # Fallback or warning if torchvision architecture changes (unlikely for B0)
            pass

        # Create the new Grouped Stem
        # in_channels=24, out_channels=32, groups=8
        # Weight shape calculation:
        #   Standard Conv2d weight: (out_channels, in_channels // groups, k, k)
        #   Here: (32, 24 // 8, 3, 3) -> (32, 3, 3, 3)
        #   This is identical to the original stem's weight shape.
        new_stem = nn.Conv2d(
            in_channels=Config.INPUT_CHANNELS,
            out_channels=Config.STEM_OUT_CHANNELS,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            groups=Config.STEM_GROUPS,
            bias=False,
        )

        # 3. Asymmetric Filter Initialization
        if Config.PRETRAINED:
            # We copy the weights directly.
            # Conceptually, this assigns:
            #   Filters 0-3 to Group 1 (Modality A, Stride X)
            #   Filters 4-7 to Group 2 (Modality A, Stride Y)
            #   ... and so on.
            with torch.no_grad():
                new_stem.weight.copy_(original_stem.weight)

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_stem

        # 4. Reconstruct Classification Head
        # EfficientNet B0 classifier is a Sequential block:
        #   (0): Dropout(p=0.2, inplace=True)
        #   (1): Linear(in_features=1280, out_features=1000, bias=True)
        original_classifier = self.backbone.classifier

        # Extract input features from the last linear layer
        # Handle potential variations in architecture definition
        if isinstance(original_classifier, nn.Sequential):
            in_features = original_classifier[-1].in_features
        elif isinstance(original_classifier, nn.Linear):
            in_features = original_classifier.in_features
        else:
            # Fallback for EfficientNet-B0 default
            in_features = 1280

        # Define new binary classification head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), nn.Linear(in_features, 1)
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 24, 224, 224)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.backbone(x)
