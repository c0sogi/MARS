import torch
import torch.nn as nn
import timm

# Import configuration from the provided library
from library.config import (
    BACKBONE,
    NUM_CLASSES,
    DROPOUT_RATE,
    IN_CHANNELS,
)


class CASIVNet(nn.Module):
    """
    Centroid-Aligned Scale-Invariant Volumetric (CASIV) Network.

    This model adapts a 2D EfficientNet backbone to process 9-channel volumetric inputs
    (3 modalities x 3 depths) by inflating the pretrained weights using a Gaussian-like
    prior.
    """

    def __init__(self):
        super(CASIVNet, self).__init__()

        # 1. Initialize Backbone
        # We use timm to load the EfficientNet-B0 with Noisy Student weights (ns)
        # drop_rate sets the dropout probability before the final classifier
        self.backbone = timm.create_model(
            BACKBONE,
            pretrained=True,
            num_classes=NUM_CLASSES,
            drop_rate=DROPOUT_RATE,
        )

        # 2. Adapt First Layer for 9-Channel Input
        self.adapt_first_layer()

    def adapt_first_layer(self):
        """
        Modifies the first convolutional layer to accept 9 channels instead of 3.
        Applies 'Gaussian Weight Inflation' to preserve ImageNet priors.
        """
        # In timm's EfficientNet implementation, the first layer is named 'conv_stem'
        old_conv = self.backbone.conv_stem

        # Extract parameters from the existing layer
        out_channels = old_conv.out_channels
        kernel_size = old_conv.kernel_size
        stride = old_conv.stride
        padding = old_conv.padding
        bias = old_conv.bias  # Usually None for conv_stem in EfficientNet

        # Create the new layer with IN_CHANNELS (9)
        new_conv = nn.Conv2d(
            in_channels=IN_CHANNELS,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias is not None,
        )

        # 3. Gaussian Weight Inflation
        # Original weights shape: (out_channels, 3, kernel_size, kernel_size)
        old_weights = old_conv.weight.data

        # New weights shape: (out_channels, 9, kernel_size, kernel_size)
        # We initialize with zeros first
        new_weights = torch.zeros_like(new_conv.weight.data)

        # Logic:
        # Channels 0-2: Peripheral (-10%) -> 25% energy
        # Channels 3-5: Center (CoM)     -> 50% energy
        # Channels 6-8: Peripheral (+10%) -> 25% energy

        # We assume the input is stacked as [Modality1_Low, Modality2_Low, Modality3_Low,
        #                                  Modality1_Mid, Modality2_Mid, Modality3_Mid, ...]
        # The original weights expect [R, G, B]. We map the 3 modalities to these 3 channels.

        # Peripheral Low (Channels 0-2)
        new_weights[:, 0:3, :, :] = old_weights * 0.25

        # Center (Channels 3-5)
        new_weights[:, 3:6, :, :] = old_weights * 0.50

        # Peripheral High (Channels 6-8)
        new_weights[:, 6:9, :, :] = old_weights * 0.25

        # Assign the new weights to the layer
        new_conv.weight.data = new_weights

        # If bias existed (unlikely for EfficientNet stem), copy it
        if bias is not None:
            new_conv.bias.data = bias

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_conv

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 9, H, W)
        Returns:
            torch.Tensor: Logits of shape (Batch, 1)
        """
        return self.backbone(x)
