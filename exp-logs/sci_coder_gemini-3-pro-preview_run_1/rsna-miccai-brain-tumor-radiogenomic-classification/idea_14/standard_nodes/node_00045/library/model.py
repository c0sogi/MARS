import torch
import torch.nn as nn
import timm
from library.config import Config


class WITSNet(nn.Module):
    """
    Weight-Inited Thick-Slab Independent Instance Network (WITS-Net).

    Architecture:
        - Backbone: EfficientNet-B0 (pretrained on ImageNet).
        - Input: 9 Channels (3x FLAIR, 3x T1wCE, 3x T2w).
        - Output: Binary Logits (for MGMT promoter methylation prediction).

    Structural Innovation:
        - Weight Inflation Initialization: Adapts pretrained RGB weights to 9-channel input
          by distributing channel energy, allowing immediate processing of volumetric slabs
          without training from scratch or using learnable adapters.
    """

    def __init__(self):
        super(WITSNet, self).__init__()

        # Initialize the EfficientNet-B0 backbone
        # drop_rate controls the dropout before the final classifier
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=Config.NUM_CLASSES,
            drop_rate=Config.DROPOUT_RATE,
        )

        # Adapt the first layer for 9-channel input using Weight Inflation
        self.modify_first_layer()

    def modify_first_layer(self):
        """
        Replaces the standard 3-channel input layer (conv_stem) with a 9-channel Conv2d layer.
        Calls inflate_weights to initialize the new layer based on pretrained weights.
        """
        # In timm's EfficientNet implementation, the first layer is named 'conv_stem'
        if not hasattr(self.backbone, "conv_stem"):
            raise AttributeError(
                f"Backbone {Config.MODEL_NAME} does not have attribute 'conv_stem'."
            )

        old_layer = self.backbone.conv_stem

        # Create a new Conv2d layer with 9 input channels
        # We preserve out_channels, kernel_size, stride, padding, and bias settings
        new_layer = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=old_layer.out_channels,
            kernel_size=old_layer.kernel_size,
            stride=old_layer.stride,
            padding=old_layer.padding,
            bias=(old_layer.bias is not None),
        )

        # Apply Weight Inflation Initialization
        self.inflate_weights(old_layer, new_layer)

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_layer

    def inflate_weights(self, old_layer, new_layer):
        """
        Implements Weight Inflation Initialization logic.

        The pretrained RGB weights (Shape: Out, 3, K, K) are distributed to the new
        9-channel weights (Shape: Out, 9, K, K) as follows:

        1. FLAIR (Channels 0-2) receives the Red channel weights / 3.
        2. T1wCE (Channels 3-5) receives the Green channel weights / 3.
        3. T2w   (Channels 6-8) receives the Blue channel weights / 3.

        Division by 3 ensures the total magnitude of activations remains consistent
        with the pretrained network's expectations.
        """
        with torch.no_grad():
            w_old = old_layer.weight.data  # Shape: (Out, 3, K, K)
            w_new = torch.zeros_like(new_layer.weight.data)  # Shape: (Out, 9, K, K)

            # Distribute weights
            # Note: w_old[:, 0:1, :, :] keeps the channel dim for broadcasting

            # FLAIR (Channels 0, 1, 2) <- Red Channel (Index 0)
            w_new[:, 0:3, :, :] = w_old[:, 0:1, :, :] / 3.0

            # T1wCE (Channels 3, 4, 5) <- Green Channel (Index 1)
            w_new[:, 3:6, :, :] = w_old[:, 1:2, :, :] / 3.0

            # T2w (Channels 6, 7, 8) <- Blue Channel (Index 2)
            w_new[:, 6:9, :, :] = w_old[:, 2:3, :, :] / 3.0

            # Assign the inflated weights to the new layer
            new_layer.weight.data = w_new

            # Copy bias if it exists
            if old_layer.bias is not None:
                new_layer.bias.data = old_layer.bias.data

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 9, Height, Width).

        Returns:
            torch.Tensor: Raw logits of shape (Batch, 1).
        """
        return self.backbone(x)
