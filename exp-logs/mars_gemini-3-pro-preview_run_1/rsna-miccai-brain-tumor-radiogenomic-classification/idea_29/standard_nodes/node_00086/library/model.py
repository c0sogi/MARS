import torch
import torch.nn as nn
import timm
from library.config import Config


class SDWIVNet(nn.Module):
    """
    Stochastic-Depth Weight-Inflated Volumetric (SD-WIV) Network.

    Uses EfficientNet-B0 as a backbone with a modified input layer to accept
    9-channel volumetric data (3 modalities x 3 depths). Implements Gaussian
    Weight Inflation for initialization and Structured Depth Dropout for regularization.
    """

    def __init__(self):
        super().__init__()

        # Initialize EfficientNet-B0 backbone
        # drop_rate controls the dropout before the final classifier layer
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=Config.NUM_CLASSES,
            drop_rate=Config.CLASSIFIER_DROPOUT,
        )

        # Modify the first convolutional layer to accept 9 channels
        # and apply Gaussian Weight Inflation
        self._modify_first_layer()

    def _modify_first_layer(self):
        """
        Replaces the first convolutional layer (stem) to accept 9 channels instead of 3.
        Initializes the new weights using the pretrained RGB weights scaled by energy factors.
        """
        # Access the first layer (conv_stem for EfficientNet in timm)
        old_layer = self.backbone.conv_stem

        # Create a new layer with 9 input channels
        # We preserve other attributes like out_channels, kernel_size, stride, padding
        new_layer = nn.Conv2d(
            in_channels=Config.NUM_CHANNELS,
            out_channels=old_layer.out_channels,
            kernel_size=old_layer.kernel_size,
            stride=old_layer.stride,
            padding=old_layer.padding,
            bias=old_layer.bias is not None,
        )

        # Gaussian Weight Inflation
        # Initialize the new weights based on the pretrained RGB weights
        old_weights = old_layer.weight.data  # Shape: (Out, 3, K, K)
        new_weights = new_layer.weight.data  # Shape: (Out, 9, K, K)

        # Clone old weights to avoid reference issues
        w_rgb = old_weights.clone()

        # Apply energy scaling factors as per SD-WIV strategy
        # Channels 0-2 (Peripheral Depth 40%): 25% energy
        new_weights[:, 0:3, :, :] = w_rgb * Config.WEIGHT_INFLATION_PERIPHERY

        # Channels 3-5 (Center Depth 50%): 50% energy
        new_weights[:, 3:6, :, :] = w_rgb * Config.WEIGHT_INFLATION_CENTER

        # Channels 6-8 (Peripheral Depth 60%): 25% energy
        new_weights[:, 6:9, :, :] = w_rgb * Config.WEIGHT_INFLATION_PERIPHERY

        # Assign the new weights and replace the layer in the backbone
        new_layer.weight.data = new_weights
        self.backbone.conv_stem = new_layer

    def forward(self, x):
        """
        Forward pass with Structured Depth Dropout during training.
        x: Input tensor of shape (Batch, 9, Height, Width)
        """
        # Structured Depth Dropout
        # Explicitly implemented in the model forward pass to force volumetric feature learning
        if self.training:
            # Randomly drop Center Depth (Channels 3-5)
            # We use an independent probability check for the center group
            if torch.rand(1).item() < Config.DEPTH_DROPOUT_PROB:
                x[:, 3:6, :, :] = 0.0

            # Randomly drop Peripheral Depths (Channels 0-2 and 6-8)
            # We use an independent probability check for the peripheral group
            if torch.rand(1).item() < Config.DEPTH_DROPOUT_PROB:
                x[:, 0:3, :, :] = 0.0
                x[:, 6:9, :, :] = 0.0

        return self.backbone(x)
