import torch
import torch.nn as nn
import timm
from library.config import Config


class SIRVEfficientNet(nn.Module):
    """
    Scale-Invariant Relative-Volumetric (SIRV) Network.

    This model utilizes an EfficientNet-B0 backbone adapted for 9-channel input
    (3 modalities x 3 relative depths). It employs a Gaussian Weight Inflation
    strategy to initialize the first convolutional layer, preserving ImageNet
    priors while enabling volumetric feature extraction.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        pretrained=True,
        num_classes=1,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        """
        Args:
            model_name (str): Name of the backbone model (default: efficientnet_b0).
            pretrained (bool): Whether to load ImageNet pre-trained weights.
            num_classes (int): Number of output classes (1 for binary classification).
            dropout_rate (float): Dropout rate for the classifier head.
        """
        super(SIRVEfficientNet, self).__init__()

        # Load the backbone model using timm
        # drop_rate sets the dropout probability before the final classifier
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate,
        )

        # Identify the first convolutional layer.
        # In timm's EfficientNet implementation, this is named 'conv_stem'.
        old_conv = self.backbone.conv_stem

        # Create a new convolutional layer with 9 input channels.
        # We preserve the original output channels, kernel size, stride, and padding.
        # The input channels correspond to:
        # [0-2]: FLAIR, T1wCE, T2w at 40% Depth (Peripheral)
        # [3-5]: FLAIR, T1wCE, T2w at 50% Depth (Center)
        # [6-8]: FLAIR, T1wCE, T2w at 60% Depth (Peripheral)
        new_conv = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        # Apply Gaussian Weight Inflation initialization
        self.inflate_weights(old_conv, new_conv)

        # Replace the first layer in the backbone
        self.backbone.conv_stem = new_conv

    def inflate_weights(self, original_conv, new_conv):
        """
        Initializes the new 9-channel convolution using Gaussian Weight Inflation.

        Strategy:
        - Center channels (3-5) receive 50% of the original RGB weight energy.
        - Peripheral channels (0-2 and 6-8) receive 25% of the original RGB weight energy.

        This ensures the model starts with a strong prior focused on the middle slice
        (analogous to standard 2D transfer learning) while allowing gradients to
        flow from the volumetric context immediately.
        """
        with torch.no_grad():
            # Original weights shape: (Out, 3, K, K)
            w_old = original_conv.weight

            # Calculate weighted components
            # Center triplet (Channels 3-5) -> 50% energy
            w_center = w_old * 0.5

            # Peripheral triplets (Channels 0-2 and 6-8) -> 25% energy
            w_periph = w_old * 0.25

            # Concatenate along the channel dimension (dim=1)
            # Order matches input tensor: [Peripheral (40%), Center (50%), Peripheral (60%)]
            w_new = torch.cat([w_periph, w_center, w_periph], dim=1)

            # Assign the new weights
            new_conv.weight.copy_(w_new)

            # Copy bias if it exists in the original layer
            if original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 9, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        return self.backbone(x)
