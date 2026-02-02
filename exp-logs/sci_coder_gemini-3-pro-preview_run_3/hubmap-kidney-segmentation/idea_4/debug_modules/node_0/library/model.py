import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from library.config import Config
from library.stain_deconv import StainDeconvolution


class StainNet(nn.Module):
    """
    Stain-Deconvolved U-Net++ with ConvNeXt-Tiny backbone.

    This model incorporates a physics-based stain deconvolution layer to project
    RGB images into Optical Density space (Hematoxylin & Eosin) before passing
    the augmented 5-channel input (RGB+HE) to a U-Net++ segmentation network.
    """

    def __init__(self):
        super(StainNet, self).__init__()

        # 1. Stain Deconvolution Layer
        # Transforms input (B, 3, H, W) -> (B, 5, H, W)
        # Adds Hematoxylin and Eosin optical density channels.
        self.stain_deconv = StainDeconvolution()

        # 2. U-Net++ with ConvNeXt-Tiny Backbone
        # We configure the encoder to accept 5 input channels.
        # SMP automatically modifies the first convolutional layer of the backbone.
        self.unet = smp.UnetPlusPlus(
            encoder_name=Config.BACKBONE,
            encoder_weights="imagenet" if Config.PRETRAINED else None,
            in_channels=Config.INPUT_CHANNELS,  # 5 channels (R, G, B, H, E)
            classes=Config.NUM_CLASSES,
            activation=None,  # Return logits for numerical stability with BCEWithLogitsLoss
            deep_supervision=Config.DEEP_SUPERVISION,  # Returns list of outputs if True
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input RGB image batch of shape (B, 3, H, W).

        Returns:
            torch.Tensor or List[torch.Tensor]:
                - If deep_supervision is False: Output logits (B, C, H, W).
                - If deep_supervision is True: List of output logits from different decoder depths.
        """
        # 1. Apply Stain Deconvolution
        # x becomes (B, 5, H, W)
        x_augmented = self.stain_deconv(x)

        # 2. Pass through U-Net++
        output = self.unet(x_augmented)

        return output
