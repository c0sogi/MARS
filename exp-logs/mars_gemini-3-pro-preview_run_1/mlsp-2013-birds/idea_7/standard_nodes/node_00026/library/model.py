import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class BirdResNet34(nn.Module):
    """
    ResNet-34 based model for Bird Species Classification.

    This architecture serves as the core for both the Teacher and Student models
    in the ensemble distillation pipeline. It uses a standard ResNet-34 backbone
    pretrained on ImageNet, with the classification head adapted for the
    specific number of bird species in the dataset.
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        """
        Initialize the model.

        Args:
            pretrained (bool): If True, loads weights pretrained on ImageNet.
                               Defaults to the value in Config.PRETRAINED.
        """
        super(BirdResNet34, self).__init__()

        # Load ResNet-34 backbone
        # We use the modern weights API compatible with torchvision >= 0.13
        if pretrained:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.resnet34(weights=weights)

        # The dataset pipeline handles replicating the single-channel spectrogram
        # to 3 channels (RGB). Therefore, the standard first convolutional layer
        # of ResNet (which expects 3 input channels) is preserved without modification.

        # Replace the final fully connected layer
        # The default ResNet-34 fc layer takes 512 input features.
        in_features = self.backbone.fc.in_features

        # Project to the number of classes (19)
        # Note: No Sigmoid activation is applied here. The training loop uses
        # BCEWithLogitsLoss, which is numerically more stable.
        self.backbone.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images.
                              Shape: (Batch_Size, 3, Height, Width)

        Returns:
            torch.Tensor: Raw logits for each class.
                          Shape: (Batch_Size, NUM_CLASSES)
        """
        return self.backbone(x)
