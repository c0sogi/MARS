import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from library.config import NUM_CHANNELS, NUM_MODALITIES, DROPOUT_RATE, DEVICE


class GroupedEfficientNetV2S(nn.Module):
    """
    Asymmetric Grouped EfficientNet-V2-S.

    Architecture:
    - Backbone: EfficientNet-V2-S (Fused-MBConv optimized).
    - Input: 12 Channels (4 Modalities x 3 Slices).
    - Stem: 12->24 Channel Conv2d with groups=4 for modality isolation.
    - Init: Asymmetric mapping of pre-trained ImageNet filters to modality groups.
    - Head: Dropout(0.5) + Linear(1).
    """

    def __init__(self):
        super().__init__()

        # 1. Load Pre-trained Backbone
        # efficientnet_v2_s uses Fused-MBConv in early stages, superior for dense data
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.backbone = efficientnet_v2_s(weights=weights)

        # 2. Surgical Stem Replacement
        # Original Stem: Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # We need to accept 12 channels but keep 24 output channels.
        # By using groups=4 (one per modality), we process each 3-channel modality independently.

        # Access the first layer of the features Sequential -> first block -> first conv
        old_conv = self.backbone.features[0][0]

        # Clone original weights: Shape is (24, 3, 3, 3)
        original_weights = old_conv.weight.data.clone()

        # Create the new grouped convolution
        # In_channels=12, Out_channels=24, Groups=4
        # Input per group = 12/4 = 3. Output per group = 24/4 = 6.
        new_conv = nn.Conv2d(
            in_channels=NUM_CHANNELS,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
            groups=NUM_MODALITIES,
        )

        # 3. Asymmetric Filter Initialization
        # The new layer's weight shape is (Out, In/Groups, K, K) -> (24, 12/4, 3, 3) -> (24, 3, 3, 3).
        # This matches the original weight shape exactly.
        # By copying directly, we assign:
        #   Filters 0-5  -> Group 0 (Modality 0: FLAIR)
        #   Filters 6-11 -> Group 1 (Modality 1: T1w)
        #   Filters 12-17 -> Group 2 (Modality 2: T1wCE)
        #   Filters 18-23 -> Group 3 (Modality 3: T2w)
        new_conv.weight.data = original_weights

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

        # 4. Regularized Head Replacement
        # Replace classifier with explicit Dropout and Linear layer
        # efficientnet_v2_s classifier is typically Sequential(Dropout, Linear)
        original_classifier = self.backbone.classifier
        in_features = original_classifier[-1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_RATE, inplace=True), nn.Linear(in_features, 1)
        )

    def forward(self, x):
        # Returns logits (unnormalized scores)
        return self.backbone(x)


def get_model():
    """
    Factory function to instantiate the model and move it to the configured device.
    """
    model = GroupedEfficientNetV2S()
    model = model.to(DEVICE)
    return model
