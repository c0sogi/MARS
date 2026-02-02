import torch
import torch.nn as nn
import timm
from library import config


class ACWIVNet(nn.Module):
    """
    Anatomically-Centric Weight-Inflated Volumetric (AC-WIV) Network.

    Adapts EfficientNet-B0 to accept 9-channel volumetric inputs (3 modalities x 3 depths)
    using a Gaussian Weight Inflation strategy to preserve ImageNet priors.
    """

    def __init__(
        self,
        backbone_name=config.BACKBONE,
        pretrained=config.PRETRAINED,
        input_channels=config.INPUT_CHANNELS,
    ):
        super(ACWIVNet, self).__init__()

        # Load the backbone with a single output class for binary classification
        # efficientnet_b0 is the default config.BACKBONE
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=1,
            drop_rate=config.DROPOUT_RATE,
        )

        # Modify the first convolutional layer to accept 9 channels
        self._modify_first_layer(input_channels)

    def _modify_first_layer(self, new_in_channels):
        """
        Replaces the first convolutional layer (conv_stem) with one that accepts
        'new_in_channels' and initializes it using Weight Inflation.
        """
        # Retrieve the existing first layer
        # In timm's EfficientNet implementation, this is named 'conv_stem'
        old_conv = self.model.conv_stem

        # Create a new Convolutional layer with the updated input channels
        # We preserve the output channels, kernel size, stride, padding, and bias settings
        new_conv = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Apply Gaussian Weight Inflation to initialize the new layer
        self._inflate_weights(old_conv, new_conv)

        # Replace the layer in the model
        self.model.conv_stem = new_conv

    def _inflate_weights(self, old_conv, new_conv):
        """
        Implements the Gaussian Weight Inflation strategy.

        Mapping:
        - Channels 0-2 (Depth z-delta): 25% of original weight energy.
        - Channels 3-5 (Depth z):       50% of original weight energy (Center/Focus).
        - Channels 6-8 (Depth z+delta): 25% of original weight energy.

        Total energy sums to 100% of original, preserving activation statistics.
        """
        with torch.no_grad():
            old_weights = old_conv.weight  # Shape: (Out, 3, K, K)

            # Ensure the old weights have 3 input channels (standard ImageNet)
            if old_weights.shape[1] != 3:
                # Fallback for non-standard backbones, though EfficientNet-B0 is standard
                nn.init.kaiming_normal_(
                    new_conv.weight, mode="fan_out", nonlinearity="relu"
                )
                return

            # Initialize the new weights
            # We assume the input is stacked as: [z-d (3ch), z (3ch), z+d (3ch)]

            # 1. Peripheral Slice (z - delta): Channels 0, 1, 2
            # Weight: 0.25
            new_conv.weight[:, 0:3, :, :] = old_weights * 0.25

            # 2. Center Slice (z): Channels 3, 4, 5
            # Weight: 0.50 (Primary focus)
            new_conv.weight[:, 3:6, :, :] = old_weights * 0.50

            # 3. Peripheral Slice (z + delta): Channels 6, 7, 8
            # Weight: 0.25
            new_conv.weight[:, 6:9, :, :] = old_weights * 0.25

            # Copy bias if it exists
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias

    def forward(self, x):
        """
        Forward pass.
        Input x: (Batch, 9, H, W)
        Output: (Batch, 1) - Logits
        """
        return self.model(x)
