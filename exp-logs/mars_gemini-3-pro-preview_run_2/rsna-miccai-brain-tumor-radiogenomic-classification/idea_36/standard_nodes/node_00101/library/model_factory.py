import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import library.config as config


class GroupedEfficientNetB0(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0.

    Reverts to the lighter B0 architecture to prevent overfitting on small datasets.
    Cite {solution_lesson_node_00100}

    Architecture:
    - Backbone: EfficientNet-B0.
    - Input: 12 Channels (4 Modalities x 3 Slices).
    - Stem: 12->32 Channel Conv2d with groups=4 for modality isolation.
    - Init: Asymmetric mapping of pre-trained ImageNet filters to modality groups.
    - Head: Dropout(0.5) + Linear(1).
    """

    def __init__(self):
        super().__init__()

        # 1. Load Pre-trained Backbone
        # Using B0 as per Lesson 100 to reduce capacity and improve generalization
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Surgical Stem Replacement
        # Original Stem: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # We need to accept 12 channels.
        # By using groups=4 (one per modality), we process each 3-channel modality independently.

        # Access the first layer of the features Sequential -> first block -> first conv
        old_conv = self.backbone.features[0][0]

        # Clone original weights: Shape is (32, 3, 3, 3)
        original_weights = old_conv.weight.data.clone()

        # Create the new grouped convolution
        # In_channels=12, Out_channels=32, Groups=4
        # Input per group = 12/4 = 3. Output per group = 32/4 = 8.
        new_conv = nn.Conv2d(
            in_channels=config.NUM_CHANNELS,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
            groups=config.NUM_MODALITIES,
        )

        # 3. Asymmetric Filter Initialization
        # The new layer's weight shape is (32, 3, 3, 3).
        # This matches the original weight shape exactly.
        # We perform a direct copy.
        new_conv.weight.data = original_weights

        # Replace the layer in the backbone
        # Cite {solution_lesson_node_00072}: Surgically replace only the conv layer
        self.backbone.features[0][0] = new_conv

        # 4. Regularized Head Replacement
        # Replace classifier with explicit Dropout and Linear layer
        original_classifier = self.backbone.classifier
        in_features = original_classifier[-1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=config.DROPOUT_RATE, inplace=True), nn.Linear(in_features, 1)
        )

    def forward(self, x):
        # Returns logits (unnormalized scores)
        return self.backbone(x)


def get_model():
    """
    Factory function to instantiate the model and move it to the configured device.
    """
    model = GroupedEfficientNetB0()
    model = model.to(config.DEVICE)
    return model
