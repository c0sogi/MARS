import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class BirdResNet(nn.Module):
    """
    ResNet-18 based model for Bird Species Classification.
    Uses standard ResNet architecture with Global Average Pooling.
    Cite solution_lesson_node_00019: Rigid adherence to pretrained input structure.
    Cite solution_lesson_node_00008: Simpler models for data-scarce regimes.
    """

    def __init__(self, pretrained=True, num_classes=Config.NUM_CLASSES):
        super(BirdResNet, self).__init__()

        # Load Pretrained ResNet-18
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
        except AttributeError:
            self.model = models.resnet18(pretrained=pretrained)

        # Replace the final fully connected layer
        # ResNet-18 final feature map depth is 512
        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrograms (Batch, 3, 224, 224)

        Returns:
            torch.Tensor: Logits (Batch, NumClasses)
        """
        return self.model(x)
