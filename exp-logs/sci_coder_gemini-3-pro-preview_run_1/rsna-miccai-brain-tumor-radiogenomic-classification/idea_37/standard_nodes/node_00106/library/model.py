import torch
import torch.nn as nn
import timm
from library.config import Config


class ARVSNet(nn.Module):
    """
    Aligned-Relative Volumetric Stack (ARVS) Network.

    Uses an EfficientNet-B0 backbone with a modified input layer to accept
    9-channel volumetric data (3 modalities x 3 depth offsets).

    Implements 'Gaussian Weight Inflation' to initialize the 9-channel weights
    from the 3-channel pretrained ImageNet weights.
    """

    def __init__(self):
        super(ARVSNet, self).__init__()

        # 1. Create Backbone
        # We use num_classes=1 for binary classification (logits)
        # drop_rate controls the dropout in the classifier head
        self.model = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=1,
            drop_rate=Config.DROPOUT_RATE,
        )

        # 2. Modify Input Layer (Gaussian Weight Inflation)
        # EfficientNet in timm typically uses 'conv_stem' as the first layer
        if not hasattr(self.model, "conv_stem"):
            raise AttributeError(
                f"Backbone {Config.BACKBONE} does not have 'conv_stem'. Check timm version/model."
            )

        old_conv = self.model.conv_stem
        old_weights = old_conv.weight.data  # Shape: (Out, 3, K, K)

        # Create new conv layer with 9 input channels
        # Maintain same output channels, kernel size, stride, padding, and bias settings
        new_conv = nn.Conv2d(
            in_channels=Config.NUM_CHANNELS,  # 9
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # 3. Initialize Weights
        # dataset.py produces channels in order:
        # [Mod1_Off1, Mod1_Off2, Mod1_Off3, Mod2_Off1, ..., Mod3_Off3]
        # Where Offsets are [-0.1, 0.0, 0.1]

        new_weights = torch.zeros_like(new_conv.weight.data)  # Shape: (Out, 9, K, K)

        # Iterate over the 9 target channels
        for i in range(Config.NUM_CHANNELS):
            # Determine which Modality (0, 1, 2) this channel belongs to
            # This maps to the R(0), G(1), B(2) channels of the pretrained weights
            mod_idx = i // len(Config.RELATIVE_OFFSETS)

            # Determine which Offset index (0, 1, 2) this is
            # 0 -> -0.1 (Peripheral), 1 -> 0.0 (Center), 2 -> +0.1 (Peripheral)
            offset_idx = i % len(Config.RELATIVE_OFFSETS)

            # Determine Scaling Factor based on Offset
            # Center (index 1) gets 50%, Periphery (indices 0, 2) gets 25%
            if offset_idx == 1:  # Center
                scale = Config.WEIGHT_INFLATION_CENTER
            else:  # Periphery
                scale = Config.WEIGHT_INFLATION_PERIPHERY

            # Copy and scale weights
            # We map Modality 0 -> Old Channel 0 (R)
            #        Modality 1 -> Old Channel 1 (G)
            #        Modality 2 -> Old Channel 2 (B)
            new_weights[:, i, :, :] = old_weights[:, mod_idx, :, :] * scale

        # Assign the inflated weights to the new layer
        new_conv.weight.data = new_weights

        # If bias existed, copy it (though usually conv_stem has no bias in EfficientNet)
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data

        # Replace the layer in the model
        self.model.conv_stem = new_conv

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, 9, H, W)
        Returns:
            logits: Output tensor of shape (Batch, 1)
        """
        return self.model(x)
