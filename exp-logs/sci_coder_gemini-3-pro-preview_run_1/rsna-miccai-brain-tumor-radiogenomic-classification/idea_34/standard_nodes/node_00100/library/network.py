import torch
import torch.nn as nn
import timm
from library.config import BACKBONE, PRETRAINED, NUM_CLASSES, DROPOUT_RATE, NUM_CHANNELS


class VRAWIVModel(nn.Module):
    """
    Verified ROI-Anchored Weight-Inflated Volumetric (V-RAWIV) Network.

    This model adapts an EfficientNet-B0 backbone to accept a 9-channel volumetric input
    (3 modalities x 3 depths). It uses Gaussian Weight Inflation to initialize the
    9-channel input layer from the 3-channel pretrained RGB weights, preserving
    feature extraction capabilities while integrating volumetric context.
    """

    def __init__(self):
        super(VRAWIVModel, self).__init__()

        # 1. Initialize Backbone
        # We create the model with num_classes, but we will override the head
        # to strictly enforce the specific Dropout configuration.
        self.model = timm.create_model(
            BACKBONE, pretrained=PRETRAINED, num_classes=NUM_CLASSES
        )

        # 2. Structural Innovation: Gaussian Weight Inflation
        self._inflate_weights()

        # 3. Custom Classifier Head
        # EfficientNet in timm uses 'classifier' as the final linear layer.
        # We replace it to ensure the exact Dropout rate (0.3) is applied.
        if hasattr(self.model, "classifier"):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=DROPOUT_RATE), nn.Linear(in_features, NUM_CLASSES)
            )
        elif hasattr(self.model, "fc"):
            # Fallback for ResNet-like architectures if config changes
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=DROPOUT_RATE), nn.Linear(in_features, NUM_CLASSES)
            )
        else:
            raise AttributeError("Could not find classifier/fc layer in backbone.")

    def _inflate_weights(self):
        """
        Modifies the first convolutional layer to accept 9 channels instead of 3.
        Initializes weights using Gaussian Weight Inflation:
        - Center channels (3-5): 50% of original RGB weights.
        - Peripheral channels (0-2, 6-8): 25% of original RGB weights.
        """
        # EfficientNet's first layer is named 'conv_stem' in timm
        if not hasattr(self.model, "conv_stem"):
            raise AttributeError(
                f"Backbone {BACKBONE} does not have 'conv_stem'. Check layer names."
            )

        old_conv = self.model.conv_stem
        old_weights = old_conv.weight.data  # Shape: (out_channels, 3, H, W)

        # Get layer properties
        out_channels = old_conv.out_channels
        kernel_size = old_conv.kernel_size
        stride = old_conv.stride
        padding = old_conv.padding
        bias = old_conv.bias

        # Create new Convolutional layer with NUM_CHANNELS input
        new_conv = nn.Conv2d(
            in_channels=NUM_CHANNELS,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias is not None,
        )

        # Initialize new weights with zeros
        new_weights = torch.zeros_like(new_conv.weight.data)

        # Apply Gaussian Weight Inflation
        # Input Tensor Structure:
        # [Depth 0.4 (0-2), Depth 0.5 (3-5), Depth 0.6 (6-8)]

        # 1. Center Slices (Channels 3, 4, 5) -> 50% Energy
        # Corresponds to Depth 0.5 (Anatomical Centroid)
        new_weights[:, 3:6, :, :] = old_weights * 0.5

        # 2. Peripheral Slices (Channels 0, 1, 2) -> 25% Energy
        # Corresponds to Depth 0.4
        new_weights[:, 0:3, :, :] = old_weights * 0.25

        # 3. Peripheral Slices (Channels 6, 7, 8) -> 25% Energy
        # Corresponds to Depth 0.6
        new_weights[:, 6:9, :, :] = old_weights * 0.25

        # Assign weights to the new layer
        new_conv.weight.data = new_weights

        # Copy bias if it exists
        if bias is not None:
            new_conv.bias.data = bias.data

        # Replace the layer in the model
        self.model.conv_stem = new_conv

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 9, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.model(x)
