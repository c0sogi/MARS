import torch
import torch.nn as nn
import timm
from library import config


class EEGEfficientNet(nn.Module):
    """
    EfficientNet-B0 architecture for classifying EEG Spectrograms.

    This model uses a pre-trained EfficientNet-B0 backbone from the `timm` library.
    It is configured to accept input tensors with dimensions defined in `config.IN_CHANNELS`
    (typically 3, representing replicated grayscale spectrograms) and outputs logits
    for the classes defined in `config.NUM_CLASSES`.
    """

    def __init__(self, pretrained=True):
        """
        Initialize the model.

        Args:
            pretrained (bool): If True, load weights pretrained on ImageNet.
        """
        super(EEGEfficientNet, self).__init__()

        # Create the EfficientNet-B0 model.
        # - 'efficientnet_b0': The specific architecture variant.
        # - pretrained: Loads ImageNet weights which helps with convergence.
        # - in_chans: Adapts the first conv layer to match input channels (3).
        # - num_classes: Replaces the final fully connected layer to output 6 logits.
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            in_chans=config.IN_CHANNELS,
            num_classes=config.NUM_CLASSES,
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, IN_CHANNELS, Height, Width).
                              Expected shape: (Batch, 3, 512, 512).

        Returns:
            torch.Tensor: Output logits of shape (Batch, NUM_CLASSES).
        """
        return self.backbone(x)
