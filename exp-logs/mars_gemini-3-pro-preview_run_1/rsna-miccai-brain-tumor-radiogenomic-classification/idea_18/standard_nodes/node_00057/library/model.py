import torch
import torch.nn as nn
import timm
from library.config import Config


class WIISNet(nn.Module):
    """
    Weight-Inflated Independent-Slab (WIIS) Network.

    Architecture:
        - Backbone: EfficientNet-B0 (Pretrained on ImageNet)
        - Input: 9 Channels (3 Modalities x 3 Slices)
        - Initialization: Weight Inflation to preserve ImageNet priors.
    """

    def __init__(self):
        super(WIISNet, self).__init__()

        # 1. Load Pretrained Backbone
        # We set num_classes to Config.NUM_CLASSES (1) to configure the head for binary classification.
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=Config.NUM_CLASSES,
            drop_rate=Config.DROPOUT_RATE,
        )

        # 2. Apply Weight Inflation to the first layer
        self._inflate_first_layer()

    def _inflate_first_layer(self):
        """
        Replaces the first convolutional layer (3 channels) with a 9-channel layer.
        Initializes the new weights by distributing the original RGB weights.
        """
        # Get the original first layer (usually named 'conv_stem' in EfficientNet)
        old_conv = self.backbone.conv_stem

        # Extract parameters from the original layer
        out_channels = old_conv.out_channels
        kernel_size = old_conv.kernel_size
        stride = old_conv.stride
        padding = old_conv.padding
        bias = old_conv.bias

        # Create the new layer with 9 input channels
        # Config.IN_CHANNELS should be 9 (3 modalities * 3 slices)
        new_conv = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=(bias is not None),
        )

        # --- Weight Inflation Logic ---
        # Original weight shape: (Out, 3, K, K)
        w_old = old_conv.weight.data

        # New weight shape: (Out, 9, K, K)
        w_new = torch.zeros(
            out_channels, Config.IN_CHANNELS, kernel_size[0], kernel_size[1]
        )

        # Distribute weights according to the WIIS strategy:
        # FLAIR (Channels 0-2) <- Red Channel (Index 0) / 3
        w_new[:, 0, :, :] = w_old[:, 0, :, :] / 3.0
        w_new[:, 1, :, :] = w_old[:, 0, :, :] / 3.0
        w_new[:, 2, :, :] = w_old[:, 0, :, :] / 3.0

        # T1wCE (Channels 3-5) <- Green Channel (Index 1) / 3
        w_new[:, 3, :, :] = w_old[:, 1, :, :] / 3.0
        w_new[:, 4, :, :] = w_old[:, 1, :, :] / 3.0
        w_new[:, 5, :, :] = w_old[:, 1, :, :] / 3.0

        # T2w (Channels 6-8) <- Blue Channel (Index 2) / 3
        w_new[:, 6, :, :] = w_old[:, 2, :, :] / 3.0
        w_new[:, 7, :, :] = w_old[:, 2, :, :] / 3.0
        w_new[:, 8, :, :] = w_old[:, 2, :, :] / 3.0

        # Assign the inflated weights to the new layer
        new_conv.weight.data = w_new

        # Copy bias if it exists
        if bias is not None:
            new_conv.bias.data = bias.data

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_conv

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch, 9, H, W)
        Returns:
            logits: Tensor of shape (Batch, 1)
        """
        return self.backbone(x)
