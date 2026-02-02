import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdClassifier(nn.Module):
    """
    BirdClassifier implements an EfficientNet-B0 backbone for multi-label bird species classification.

    It adapts the standard ImageNet-pretrained architecture to accept 1-channel Log-Mel Spectrograms
    and outputs logits for 19 species.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED,
        in_channels=Config.IN_CHANNELS,
    ):
        """
        Args:
            model_name (str): Name of the model architecture (e.g., 'efficientnet_b0').
            num_classes (int): Number of output classes (19).
            pretrained (bool): Whether to load ImageNet pretrained weights.
            in_channels (int): Number of input channels (1 for mono spectrograms).
        """
        super(BirdClassifier, self).__init__()

        # Initialize the model using timm
        # in_chans=1 triggers the library to modify the first convolutional layer
        # to accept single-channel input, recycling pretrained weights via averaging.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
            global_pool="avg",
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, 1, n_mels, time).

        Returns:
            torch.Tensor: Logits of shape (Batch_Size, num_classes).
        """
        return self.backbone(x)
