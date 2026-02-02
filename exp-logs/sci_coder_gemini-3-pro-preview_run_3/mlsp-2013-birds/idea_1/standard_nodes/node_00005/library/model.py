import torch
import torch.nn as nn
import torchvision.models as models
from library import config


class BirdResNet(nn.Module):
    """
    ResNet18-based model for Bird Species Classification using Spectrograms.
    Cite solution_lesson_node_00003: Transfer learning with CNNs.
    """

    def __init__(self, num_classes=config.NUM_CLASSES, pretrained=True):
        super(BirdResNet, self).__init__()

        # Load pre-trained ResNet18
        # Using weights parameter if available (torchvision >= 0.13), else pretrained
        try:
            from torchvision.models import ResNet18_Weights

            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = models.resnet18(weights=weights)
        except ImportError:
            self.resnet = models.resnet18(pretrained=pretrained)

        # Replace the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
