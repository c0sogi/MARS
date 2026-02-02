import torch
import torch.nn as nn
import timm
from library import config


class InputDepthDropout(nn.Module):
    """
    Custom module to randomly zero out specific depth triplets during training.

    Note: In this pipeline, the depth dropout logic is primarily handled within the
    data loader (library/data.py) to ensure correct caching and visualization.
    This module is included for architectural completeness.
    """

    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        # If the data loader has already applied dropout, we should not apply it again here.
        # We return x unmodified to avoid double-dropout.
        # If one wished to move the logic to the GPU, the implementation would go here.
        return x


def gaussian_weight_inflation(new_conv, old_weights):
    """
    Initializes the new 9-channel convolution weights using a Gaussian prior
    derived from the original 3-channel ImageNet weights.

    Strategy:
    - Center Slices (Channels 3-5): 50% of original weight energy.
    - Peripheral Slices (Channels 0-2, 6-8): 25% of original weight energy.
    """
    with torch.no_grad():
        # old_weights shape: (Out, 3, K, K)
        # new_conv.weight shape: (Out, 9, K, K)

        center_factor = config.WEIGHT_INIT_CENTER
        periph_factor = config.WEIGHT_INIT_PERIPHERAL

        # Initialize Peripheral (40% depth) -> Channels 0, 1, 2
        new_conv.weight[:, 0:3, :, :] = old_weights * periph_factor

        # Initialize Center (50% depth) -> Channels 3, 4, 5
        new_conv.weight[:, 3:6, :, :] = old_weights * center_factor

        # Initialize Peripheral (60% depth) -> Channels 6, 7, 8
        new_conv.weight[:, 6:9, :, :] = old_weights * periph_factor


class SIRVEfficientNet(nn.Module):
    """
    Scale-Invariant Relative-Volumetric (SIRV) Network.

    Uses an EfficientNet-B0 backbone with a modified input stem to accept
    9 channels (3 modalities x 3 relative depths).
    """

    def __init__(self):
        super().__init__()

        # Create backbone
        # drop_rate sets the dropout rate before the final classifier layer
        self.backbone = timm.create_model(
            config.BACKBONE,
            pretrained=config.PRETRAINED,
            num_classes=config.NUM_CLASSES,
            drop_rate=config.DROPOUT_RATE,
        )

        # Adjust the first convolutional layer (stem)
        # EfficientNet-B0 in timm uses 'conv_stem' for the first layer
        if hasattr(self.backbone, "conv_stem"):
            old_conv = self.backbone.conv_stem

            # Create new convolution with 9 input channels
            new_conv = nn.Conv2d(
                in_channels=config.NUM_CHANNELS,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

            # Apply Gaussian Weight Inflation
            gaussian_weight_inflation(new_conv, old_conv.weight)

            # Replace the layer
            self.backbone.conv_stem = new_conv
        else:
            # Fallback or error if backbone architecture changes
            raise AttributeError(
                f"Backbone {config.BACKBONE} does not have 'conv_stem'."
            )

        # Input Depth Dropout Module
        self.depth_dropout = InputDepthDropout(p=config.DEPTH_DROPOUT_PROB)

    def forward(self, x):
        # Apply Input Depth Dropout (Identity if handled by loader)
        x = self.depth_dropout(x)

        # Forward pass through backbone
        x = self.backbone(x)

        return x
