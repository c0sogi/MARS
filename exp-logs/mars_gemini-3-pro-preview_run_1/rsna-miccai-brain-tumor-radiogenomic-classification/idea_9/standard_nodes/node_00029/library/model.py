import torch
import torch.nn as nn
import timm
from library.config import Config


class MGMTClassifier(nn.Module):
    """
    MGMT Promoter Methylation Classifier.

    This class implements the neural network architecture based on EfficientNet-B0.
    It is designed to process 3-channel 2D inputs (FLAIR, T1wCE, T2w) and output
    a single logit for binary classification.
    """

    def __init__(self):
        super(MGMTClassifier, self).__init__()

        # Initialize the EfficientNet-B0 backbone using timm
        # Config.MODEL_NAME: "efficientnet_b0"
        # Config.PRETRAINED: True (Load ImageNet weights)
        # Config.NUM_CLASSES: 1 (Binary classification logit)
        # Config.NUM_CHANNELS: 3 (FLAIR, T1wCE, T2w)
        # We rely on default dropout rates as per the lesson to avoid regression.
        self.model = timm.create_model(
            model_name=Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=Config.NUM_CLASSES,
            in_chans=Config.NUM_CHANNELS,
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, 3, 224, 224).

        Returns:
            torch.Tensor: Output logits of shape (Batch_Size, 1).
        """
        return self.model(x)
