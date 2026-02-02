import torch
import torch.nn as nn
import timm
from library.config import (
    INPUT_CHANNELS,
    NUM_CLASSES,
    DROPOUT_RATE,
    INPUT_DROPOUT_PROB,
    BACKBONE,
)


class RARVEfficientNet(nn.Module):
    def __init__(self):
        super(RARVEfficientNet, self).__init__()

        # Initialize the backbone using timm
        # efficientnet_b0 is used as per configuration
        self.backbone = timm.create_model(
            BACKBONE, pretrained=True, num_classes=NUM_CLASSES, drop_rate=DROPOUT_RATE
        )

        # Adapt the first layer for 9-channel input
        self.modify_first_layer()

    def modify_first_layer(self):
        """
        Replaces the first convolutional layer (conv_stem) to accept 9 input channels.
        Initializes weights using Gaussian Weight Inflation:
        - Center channels (3-5) get 50% of original weights.
        - Peripheral channels (0-2, 6-8) get 25% of original weights.
        """
        old_conv = self.backbone.conv_stem
        old_weights = old_conv.weight.data  # Shape: (Out, 3, K, K)

        # Create new convolutional layer
        # Check if bias exists (EfficientNet usually has bias=False in stem)
        use_bias = old_conv.bias is not None

        new_conv = nn.Conv2d(
            in_channels=INPUT_CHANNELS,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=use_bias,
        )

        # Initialize new weights with zeros
        new_weights = torch.zeros_like(new_conv.weight.data)  # Shape: (Out, 9, K, K)

        # Apply Gaussian Weight Inflation
        # Channels 0-2: Peripheral (Depth 0.4) -> 25% energy
        new_weights[:, 0:3, :, :] = old_weights * 0.25

        # Channels 3-5: Center (Depth 0.5) -> 50% energy
        new_weights[:, 3:6, :, :] = old_weights * 0.50

        # Channels 6-8: Peripheral (Depth 0.6) -> 25% energy
        new_weights[:, 6:9, :, :] = old_weights * 0.25

        # Assign weights to the new layer
        new_conv.weight.data = new_weights

        # Copy bias if it existed
        if use_bias:
            new_conv.bias.data = old_conv.bias.data

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_conv

    def structured_input_dropout(self, x):
        """
        Applies structured dropout to input channels during training.
        Randomly zeros out either the center triplet or the peripheral triplets.
        """
        if not self.training:
            return x

        # Check if we should apply dropout based on probability
        if torch.rand(1).item() < INPUT_DROPOUT_PROB:
            # 50% chance to drop Center vs Peripherals
            if torch.rand(1).item() < 0.5:
                # Drop Center Triplet (Channels 3, 4, 5)
                x[:, 3:6, :, :] = 0
            else:
                # Drop Peripheral Triplets (Channels 0, 1, 2 and 6, 7, 8)
                x[:, 0:3, :, :] = 0
                x[:, 6:9, :, :] = 0

        return x

    def forward(self, x):
        # Apply Structured Input Dropout
        x = self.structured_input_dropout(x)

        # Pass through the backbone
        # The backbone includes the global pooling and classifier head
        x = self.backbone(x)

        return x
