import torch
import torch.nn as nn
import timm
from library.config import Config


class SILNet(nn.Module):
    """
    Stratified Instance-Level 2.5D Network (SIL-Net).

    This architecture uses a standard EfficientNet-B0 backbone initialized with
    ImageNet weights. It is designed to process 3-channel 2D composite images
    constructed from FLAIR, T1wCE, and T2w MRI slices.

    The network outputs a single logit representing the probability of MGMT
    promoter methylation.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        in_chans=Config.CHANNELS,
    ):
        """
        Initialize the SILNet model.

        Args:
            model_name (str): Name of the backbone architecture (default: 'efficientnet_b0').
            pretrained (bool): Whether to load pretrained ImageNet weights (default: True).
            num_classes (int): Number of output classes (default: 1 for binary classification).
            in_chans (int): Number of input channels (default: 3).
        """
        super(SILNet, self).__init__()

        # Create the backbone model using timm
        # This automatically handles the modification of the classifier head
        # to match num_classes.
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=Config.DROP_RATE,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, Height, Width).
                              Expected shape is (B, 3, 224, 224).

        Returns:
            torch.Tensor: Output logits of shape (Batch_Size, 1).
        """
        return self.model(x)
