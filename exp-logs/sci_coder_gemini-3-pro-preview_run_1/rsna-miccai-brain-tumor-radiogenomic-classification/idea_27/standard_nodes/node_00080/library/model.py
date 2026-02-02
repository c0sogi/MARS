import torch
import torch.nn as nn
import timm
from library.config import DEVICE


class SIRVEfficientNet(nn.Module):
    """
    Scale-Invariant Relative-Volumetric (SIRV) Network.

    This model utilizes an EfficientNet-B0 backbone adapted for 9-channel input
    (3 MRI modalities x 3 relative anatomical depths). It implements a custom
    Gaussian Weight Inflation initialization strategy to preserve ImageNet priors
    while integrating volumetric context.
    """

    def __init__(self, model_name="efficientnet_b0", pretrained=True, dropout_rate=0.3):
        """
        Args:
            model_name (str): Name of the timm model to load.
            pretrained (bool): Whether to load ImageNet weights.
            dropout_rate (float): Dropout rate for the classifier head.
        """
        super(SIRVEfficientNet, self).__init__()

        # 1. Load Backbone
        # Initialize standard EfficientNet-B0 with 3 input channels.
        # We set num_classes=1 for the final binary target.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=1, drop_rate=dropout_rate
        )

        # 2. Modify First Layer (3 channels -> 9 channels)
        self._modify_first_layer()

        # 3. Enforce Classifier Structure
        # Explicitly define the classifier to ensure the specific dropout requirement is met.
        # In timm's EfficientNet, the final layer is usually named 'classifier'.
        if hasattr(self.backbone, "classifier"):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate), nn.Linear(in_features, 1)
            )

        # Move model to the configured device
        self.to(DEVICE)

    def _modify_first_layer(self):
        """
        Replaces the first convolutional layer (conv_stem) to accept 9 input channels.
        Initializes weights using Gaussian Weight Inflation:
        - Center slices (channels 3-5) get 50% of original weight energy.
        - Peripheral slices (channels 0-2, 6-8) get 25% of original weight energy.
        """
        # Retrieve the original first layer (Standard RGB input)
        old_conv = self.backbone.conv_stem
        old_weights = old_conv.weight.data  # Shape: (Out, 3, K, K)

        # Create new layer with 9 input channels
        # Structure: [Depth1_Mods, Depth2_Mods, Depth3_Mods]
        new_conv = nn.Conv2d(
            in_channels=9,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Initialize new weights tensor
        new_weights = torch.zeros_like(new_conv.weight.data)  # Shape: (Out, 9, K, K)

        # Apply Gaussian Weight Inflation
        # 1. Peripheral Depth (40%): Channels 0, 1, 2 -> 25% Energy
        new_weights[:, 0:3, :, :] = old_weights * 0.25

        # 2. Center Depth (50%): Channels 3, 4, 5 -> 50% Energy
        # This preserves the strong "middle slice" baseline behavior.
        new_weights[:, 3:6, :, :] = old_weights * 0.50

        # 3. Peripheral Depth (60%): Channels 6, 7, 8 -> 25% Energy
        new_weights[:, 6:9, :, :] = old_weights * 0.25

        # Assign weights to the new layer
        new_conv.weight.data = new_weights

        # Copy bias if it exists
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_conv

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 9, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        return self.backbone(x)
