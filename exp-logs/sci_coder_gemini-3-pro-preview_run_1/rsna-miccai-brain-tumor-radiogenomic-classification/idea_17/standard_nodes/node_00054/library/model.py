import torch
import torch.nn as nn
import timm
from library.config import Config


class AAWIISNet(nn.Module):
    """
    Anatomically-Anchored Weight-Inflated 2.5D Slab Network (AA-WIIS-Net).

    This model utilizes an EfficientNet-B0 backbone modified to accept a 9-channel
    input tensor. The 9 channels represent 3 consecutive slices from 3 different
    MRI modalities (FLAIR, T1wCE, T2w).

    To leverage ImageNet pre-training without destroying feature detectors,
    the first convolutional layer is 'inflated' by replicating the original RGB
    weights and scaling them to preserve signal energy.
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        """
        Args:
            pretrained (bool): Whether to load ImageNet pre-trained weights.
        """
        super(AAWIISNet, self).__init__()

        # Initialize EfficientNet-B0 backbone
        # num_classes=1 for binary classification (output is a logit)
        # drop_rate controls the dropout before the final classifier
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=pretrained,
            num_classes=1,
            drop_rate=Config.DROPOUT_RATE,
        )

        # Modify the first layer to handle 9 channels via Weight Inflation
        self._inflate_weights()

    def _inflate_weights(self):
        """
        Replaces the first convolutional layer (conv_stem) with a layer capable
        of handling Config.NUM_CHANNELS (9) inputs.

        The weights are initialized by repeating the original 3-channel (RGB) weights
        3 times along the channel dimension and dividing by 3. This maps the
        RGB pattern to each modality triplet (FLAIR, T1wCE, T2w) equally.
        """
        # Retrieve the original first layer
        old_conv = self.backbone.conv_stem

        # Create a new Conv2d layer with the target number of input channels
        # We preserve the original output channels, kernel size, stride, and padding
        new_conv = nn.Conv2d(
            in_channels=Config.NUM_CHANNELS,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Perform Weight Inflation
        with torch.no_grad():
            if old_conv.weight is not None:
                # Original weight shape: (Out, 3, H, W)
                old_weights = old_conv.weight

                # Repeat weights 3 times along the channel dimension (dim 1)
                # New shape: (Out, 9, H, W)
                # This effectively copies the RGB filters to channels 0-2, 3-5, and 6-8
                new_weights = old_weights.repeat(1, 3, 1, 1)

                # Scale weights by 1/3 to preserve the expected activation variance
                # (Energy conservation principle for sum of inputs)
                new_weights = new_weights / 3.0

                # Assign inflated weights to the new layer
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
            x (torch.Tensor): Input tensor of shape (Batch_Size, 9, H, W).

        Returns:
            torch.Tensor: Output logits of shape (Batch_Size, 1).
        """
        return self.backbone(x)
