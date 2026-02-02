import torch
import torch.nn as nn
import timm
from library import config


class BirdCNN(nn.Module):
    """
    CNN for Bird Species Classification using Spectrograms.
    Uses a pre-trained ResNet18 backbone.
    """

    def __init__(
        self, model_name="resnet18", num_classes=config.NUM_CLASSES, pretrained=True
    ):
        super(BirdCNN, self).__init__()

        # Load pre-trained model
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)
