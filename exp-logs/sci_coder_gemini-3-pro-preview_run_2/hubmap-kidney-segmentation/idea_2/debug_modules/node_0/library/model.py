import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from library.config import Config


class AnatomyAwareUNetPlusPlus(nn.Module):
    """
    Anatomy-Aware U-Net++ Model.

    This architecture uses a ResNet-34 encoder within a U-Net++ structure.
    It is specifically modified to accept 4-channel inputs:
    1-3. RGB Image Channels
    4.   Binary Anatomical Mask (Cortex vs Medulla/Background)

    The anatomical mask provides explicit spatial context to the encoder,
    acting as a hard spatial attention mechanism to guide the detection
    of functional tissue units.
    """

    def __init__(self):
        """
        Initializes the model, loads pre-trained ImageNet weights for the encoder,
        and adapts the input layer to accept 4 channels.
        """
        super(AnatomyAwareUNetPlusPlus, self).__init__()

        # Instantiate the U-Net++ model from segmentation_models_pytorch.
        # We initialize with in_channels=3 first to ensure the library loads
        # the standard ImageNet pre-trained weights for the ResNet encoder.
        # activation=None ensures we return logits, which is required by
        # the HybridBCEDiceLoss (which applies sigmoid internally).
        self.model = smp.UnetPlusPlus(
            encoder_name=Config.ENCODER_NAME,
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=3,
            classes=Config.CLASSES,
            activation=None,
        )

        # Modify the encoder to handle 4 channels (RGB + Anatomy)
        self._adapt_input_layer()

    def _adapt_input_layer(self):
        """
        Replaces the first convolutional layer of the encoder to accept
        Config.IN_CHANNELS (4) instead of the default 3.
        """
        # Access the first conv layer. For ResNet encoders in SMP, this is 'conv1'.
        old_conv = self.model.encoder.conv1

        # Create a new Conv2d layer with the updated input channels
        # We preserve all other properties (out_channels, kernel, stride, padding)
        new_conv = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Weight Initialization Strategy:
        # 1. Copy the pre-trained ImageNet weights to the first 3 channels (RGB).
        # 2. Initialize the 4th channel (Anatomical Mask) with the mean of the RGB weights.
        #    This allows the new channel to contribute to features with a similar
        #    magnitude distribution as the visual features at the start of training,
        #    preventing a 'cold start' for the extra channel.
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:4, :, :] = torch.mean(
                old_conv.weight, dim=1, keepdim=True
            )

            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias

        # Replace the layer in the encoder
        self.model.encoder.conv1 = new_conv

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of shape (N, 4, H, W).
                              Channels are [R, G, B, Anatomical_Mask].

        Returns:
            torch.Tensor: Output logits of shape (N, 1, H, W).
        """
        return self.model(x)
