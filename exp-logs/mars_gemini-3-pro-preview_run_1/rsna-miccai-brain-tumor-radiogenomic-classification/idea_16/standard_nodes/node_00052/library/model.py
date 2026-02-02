import torch
import torch.nn as nn
import timm
from library.config import Config


class WIISNet(nn.Module):
    """
    Weight-Inflated Independent-Slab Network (WIIS-Net).

    This architecture uses an EfficientNet-B0 backbone adapted for 9-channel input
    (3 modalities * 3 slices/slab). It employs a custom 'Weight Inflation' initialization
    strategy to preserve ImageNet priors by distributing RGB weights across the
    corresponding modality channels.
    """

    def __init__(self):
        super(WIISNet, self).__init__()

        # Load the pretrained EfficientNet-B0 backbone
        # We use the hyperparameters defined in Config
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=Config.NUM_CLASSES,
            drop_rate=Config.DROPOUT_RATE,
        )

        # Apply the Weight Inflation Initialization to the first layer
        self._inflate_first_layer()

    def _inflate_first_layer(self):
        """
        Replaces the first convolutional layer (3 input channels) with a new layer
        (9 input channels) and initializes it by distributing the original RGB weights.
        """
        # In timm's EfficientNet implementation, the first layer is named 'conv_stem'
        old_conv = self.backbone.conv_stem

        # Extract configuration from the existing layer
        out_channels = old_conv.out_channels
        kernel_size = old_conv.kernel_size
        stride = old_conv.stride
        padding = old_conv.padding
        bias = old_conv.bias

        # Create the new layer with the target number of input channels (9)
        new_conv = nn.Conv2d(
            in_channels=Config.INPUT_CHANNELS,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=(bias is not None),
        )

        # --- Weight Inflation Logic ---
        # Original weights shape: (Out, 3, K, K)
        # New weights shape:      (Out, 9, K, K)
        old_weights = old_conv.weight.data
        new_weights = torch.zeros_like(new_conv.weight.data)

        # Mapping:
        # RGB Red   (Idx 0) -> FLAIR Slab (Idx 0, 1, 2)
        # RGB Green (Idx 1) -> T1wCE Slab (Idx 3, 4, 5)
        # RGB Blue  (Idx 2) -> T2w Slab   (Idx 6, 7, 8)

        # We divide by 3.0 so that the sum of weights for a constant input remains
        # roughly equivalent to the original single-channel response.

        # 1. Distribute Red weights to FLAIR channels
        new_weights[:, 0:3, :, :] = old_weights[:, 0:1, :, :] / 3.0

        # 2. Distribute Green weights to T1wCE channels
        new_weights[:, 3:6, :, :] = old_weights[:, 1:2, :, :] / 3.0

        # 3. Distribute Blue weights to T2w channels
        new_weights[:, 6:9, :, :] = old_weights[:, 2:3, :, :] / 3.0

        # Assign the inflated weights to the new layer
        new_conv.weight.data = new_weights

        # Copy bias if present (EfficientNet conv_stem usually has no bias due to BN, but we handle it)
        if bias is not None:
            new_conv.bias.data = old_conv.bias.data

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_conv

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 9, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        return self.backbone(x)
