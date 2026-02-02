import torch
import torch.nn as nn
import timm
from library.config import IN_CHANNELS


class AGIVEfficientNet(nn.Module):
    """
    Anatomically-Anchored Gaussian-Inflated Volumetric (AGIV) Network.

    Uses EfficientNet-B0 as a backbone, modified to accept a 9-channel volumetric input.
    The first convolutional layer is initialized using a Gaussian-like profile along the
    z-axis (channel groups) to preserve ImageNet priors while integrating spatial context.
    """

    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        super(AGIVEfficientNet, self).__init__()

        # Initialize the backbone with 1 output class for binary classification (logits)
        # efficientnet_b0 is chosen for its efficiency and performance on this dataset size
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=1
        )

        # Modify the input layer to handle 9 channels with specific weight initialization
        self._modify_first_conv_layer()

    def _modify_first_conv_layer(self):
        """
        Replaces the first convolutional layer (conv_stem) to accept IN_CHANNELS (9).
        Initializes weights using the Gaussian Weight Inflation strategy.
        """
        # Access the original first layer (usually named 'conv_stem' in timm EfficientNets)
        old_conv = self.backbone.conv_stem

        # Create the new layer with the target input channels
        # We preserve the original output channels, kernel size, stride, and padding
        new_conv = nn.Conv2d(
            in_channels=IN_CHANNELS,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # --- Gaussian Weight Inflation Initialization ---
        # The input tensor is packed as:
        # [Channels 0-2: z-delta], [Channels 3-5: z (Center)], [Channels 6-8: z+delta]

        # 1. Get original ImageNet weights: Shape [Out, 3, K, K]
        w_rgb = old_conv.weight.data

        # 2. Calculate weights for the Center Slices (strongest prior)
        # Receives 50% of the original weight energy
        w_center = w_rgb * 0.5

        # 3. Calculate weights for the Peripheral Slices (context)
        # Receives 25% of the original weight energy
        w_periph = w_rgb * 0.25

        # 4. Construct the new weight tensor by concatenating along the input channel dimension (dim=1)
        # Order: [Periph (z-delta), Center (z), Periph (z+delta)]
        w_new = torch.cat([w_periph, w_center, w_periph], dim=1)

        # 5. Assign the new weights
        new_conv.weight.data = w_new

        # Copy bias if it exists (EfficientNet stem usually has no bias, but for safety)
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_conv

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 9, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.backbone(x)


def build_model(device):
    """
    Factory function to build and move the model to the specified device.
    """
    model = AGIVEfficientNet()
    model.to(device)
    return model
