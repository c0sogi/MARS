import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdCNN(nn.Module):
    """
    A wrapper around timm models for the Deep Learning Stream (CNNs).
    Supports ResNet-18, EfficientNet-B0, and DenseNet-121.
    """

    def __init__(self, model_name, pretrained=True):
        """
        Args:
            model_name (str): Name of the architecture (e.g., 'resnet18').
            pretrained (bool): Whether to load ImageNet pre-trained weights.
        """
        super(BirdCNN, self).__init__()

        # Create the model using timm
        # in_chans=3 because we replicate the grayscale spectrogram to 3 channels
        # num_classes=Config.NUM_CLASSES (19)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=Config.NUM_CLASSES,
            in_chans=Config.INPUT_CHANNELS,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, NUM_CLASSES)
        """
        return self.backbone(x)
