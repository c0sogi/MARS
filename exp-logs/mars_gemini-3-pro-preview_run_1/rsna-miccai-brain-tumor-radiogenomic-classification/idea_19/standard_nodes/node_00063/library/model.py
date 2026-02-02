import torch
import torch.nn as nn
import timm
from library import config


class WIVENet(nn.Module):
    """
    Weight-Inflated Volumetric Early-Fusion (WIVE) Network.

    Wraps an EfficientNet-B0 backbone and adapts the first convolutional layer
    to accept 9-channel volumetric inputs (3 depths x 3 modalities) by inflating
    pretrained ImageNet weights.
    """

    def __init__(self):
        super(WIVENet, self).__init__()

        # 1. Load Pretrained Backbone (Standard 3-channel init to get ImageNet weights)
        self.backbone = timm.create_model(
            config.MODEL_NAME,
            pretrained=True,
            num_classes=config.NUM_CLASSES,
            drop_rate=config.DROPOUT_RATE,
        )

        # 2. Perform Weight Inflation on the Stem
        self._inflate_first_layer()

    def _inflate_first_layer(self):
        """
        Replaces the first convolutional layer (conv_stem) with a 9-channel version.
        Weights are initialized by distributing the original RGB weights across
        the corresponding modality channels at different depths, scaled by 1/3.
        """
        # Retrieve the existing layer
        old_layer = self.backbone.conv_stem

        # Define new input channels (9)
        new_in_channels = config.IN_CHANNELS

        # Create the new layer with target dimensions but same kernel/stride/padding
        new_layer = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=old_layer.out_channels,
            kernel_size=old_layer.kernel_size,
            stride=old_layer.stride,
            padding=old_layer.padding,
            bias=(old_layer.bias is not None),
        )

        # --- Weight Inflation Logic ---
        # Original shape: (Out, 3, K, K)
        old_weights = old_layer.weight.data
        # New shape: (Out, 9, K, K)
        new_weights = torch.zeros_like(new_layer.weight.data)

        # Mapping Strategy:
        # Input Tensor Structure:
        # [0]: Depth A - FLAIR  -> Map to Red
        # [1]: Depth A - T1wCE  -> Map to Green
        # [2]: Depth A - T2w    -> Map to Blue
        # [3]: Depth B - FLAIR  -> Map to Red
        # [4]: Depth B - T1wCE  -> Map to Green
        # [5]: Depth B - T2w    -> Map to Blue
        # [6]: Depth C - FLAIR  -> Map to Red
        # [7]: Depth C - T1wCE  -> Map to Green
        # [8]: Depth C - T2w    -> Map to Blue

        # We divide by 3.0 because we are summing over 3 depth slices.
        # This preserves the expected activation magnitude.

        # 1. FLAIR (Channels 0, 3, 6) <- Red (Channel 0)
        new_weights[:, [0, 3, 6], :, :] = old_weights[:, 0:1, :, :] / 3.0

        # 2. T1wCE (Channels 1, 4, 7) <- Green (Channel 1)
        new_weights[:, [1, 4, 7], :, :] = old_weights[:, 1:2, :, :] / 3.0

        # 3. T2w (Channels 2, 5, 8) <- Blue (Channel 2)
        new_weights[:, [2, 5, 8], :, :] = old_weights[:, 2:3, :, :] / 3.0

        # Assign weights
        new_layer.weight.data = new_weights

        # Copy bias if it exists
        if old_layer.bias is not None:
            new_layer.bias.data = old_layer.bias.data

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_layer

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 9, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.backbone(x)
