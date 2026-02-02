import torch
import torch.nn as nn
import timm
from library.config import Config


class EfficientNetExpert(nn.Module):
    """
    EfficientNet-B0 based Expert Model for the VCAE strategy.

    This model is designed to process a specific anatomical plane (anchored to the
    Brain Center of Mass) represented as a 3-channel image (FLAIR, T1wCE, T2w).

    It wraps the timm implementation of EfficientNet-B0, adapting the input layer
    to the correct number of channels and the output layer for binary classification.
    """

    def __init__(self, pretrained=True):
        """
        Initialize the EfficientNetExpert.

        Args:
            pretrained (bool): If True, loads weights pretrained on ImageNet.
                               Default is True.
        """
        super(EfficientNetExpert, self).__init__()

        # Load EfficientNet-B0 from timm
        # in_chans=Config.NUM_CHANNELS (3) ensures the first convolution layer
        # matches the input modalities [FLAIR, T1wCE, T2w].
        # num_classes=1 sets the final fully connected layer to output a single logit.
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=1,
            in_chans=Config.NUM_CHANNELS,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, 3, 224, 224).

        Returns:
            torch.Tensor: Raw logits of shape (Batch_Size, 1).
                          Sigmoid activation should be applied externally (e.g., in Loss or Inference).
        """
        return self.backbone(x)
