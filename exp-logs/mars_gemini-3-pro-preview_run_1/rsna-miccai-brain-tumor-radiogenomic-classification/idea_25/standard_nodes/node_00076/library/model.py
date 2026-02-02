import torch
import torch.nn as nn
import timm
from library import config


class EfficientNet9Channel(nn.Module):
    """
    EfficientNet-B0 modified to accept 9-channel volumetric inputs.
    Implements 'Gaussian Weight Inflation' for initialization to preserve
    ImageNet priors while integrating volumetric context.
    """

    def __init__(self, backbone_name=config.BACKBONE, pretrained=True, num_classes=1):
        """
        Args:
            backbone_name (str): Name of the timm model to use.
            pretrained (bool): Whether to load ImageNet weights.
            num_classes (int): Number of output classes (1 for binary classification).
        """
        super(EfficientNet9Channel, self).__init__()

        # Create the base model using timm
        # num_classes=1 sets up the classifier for binary output (logits)
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=num_classes
        )

        # Modify the first layer to accept 9 channels
        self._modify_first_layer()

    def _modify_first_layer(self):
        """
        Replaces the first convolutional layer (conv_stem) with a 9-channel version
        and applies the custom weight initialization.
        """
        # Access the original first layer
        original_conv = self.backbone.conv_stem

        # Create a new Conv2d layer with 9 input channels
        # We preserve the original output channels, kernel size, stride, etc.
        new_conv = nn.Conv2d(
            in_channels=config.INPUT_CHANNELS,  # 9
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None),
        )

        # Initialize the new layer's weights using the Gaussian Weight Inflation strategy
        self._adapt_weights(original_conv, new_conv)

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_conv

    def _adapt_weights(self, original_conv, new_conv):
        """
        Adapts the pre-trained 3-channel weights to the new 9-channel layer.

        Strategy:
        - Center Slices (Channels 3-5): 50% of original weight energy.
        - Peripheral Slices (Channels 0-2, 6-8): 25% of original weight energy.

        This sums to 100% (0.25 + 0.5 + 0.25), preserving the magnitude of activations
        expected by the subsequent layers.
        """
        with torch.no_grad():
            old_weights = original_conv.weight.data  # Shape: (Out, 3, K, K)
            new_weights = new_conv.weight.data  # Shape: (Out, 9, K, K)

            # Ensure we are modifying the tensor in-place or assigning correctly

            # 1. Peripheral Slices (z - delta): Channels 0, 1, 2
            # Scale factor: 0.25
            new_weights[:, 0:3, :, :] = old_weights * 0.25

            # 2. Center Slices (z): Channels 3, 4, 5
            # Scale factor: 0.50 (Stronger prior for the anatomical center)
            new_weights[:, 3:6, :, :] = old_weights * 0.50

            # 3. Peripheral Slices (z + delta): Channels 6, 7, 8
            # Scale factor: 0.25
            new_weights[:, 6:9, :, :] = old_weights * 0.25

            # Assign back to the new layer
            new_conv.weight.data = new_weights

            # Copy bias if it exists
            if original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 9, H, W).
        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        return self.backbone(x)
