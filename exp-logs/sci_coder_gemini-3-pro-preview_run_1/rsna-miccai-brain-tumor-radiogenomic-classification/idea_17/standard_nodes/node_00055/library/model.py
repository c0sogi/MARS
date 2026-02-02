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
        # We use standard 3-channel input (RGB) to leverage pretraining stability (Cite solution_lesson_node_00009)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=pretrained,
            num_classes=1,
            drop_rate=Config.DROPOUT_RATE,
            in_chans=Config.NUM_CHANNELS,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, 9, H, W).

        Returns:
            torch.Tensor: Output logits of shape (Batch_Size, 1).
        """
        return self.backbone(x)
