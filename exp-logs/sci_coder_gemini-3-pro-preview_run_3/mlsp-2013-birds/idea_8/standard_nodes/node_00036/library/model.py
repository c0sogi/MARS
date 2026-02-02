import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class BirdResNet(nn.Module):
    """
    Standard ResNet-18 model for bird species classification.
    """

    def __init__(self):
        super(BirdResNet, self).__init__()

        # Load Pretrained ResNet18
        weights = "DEFAULT" if Config.PRETRAINED else None
        self.backbone = models.resnet18(weights=weights)

        # Modify the final Fully Connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, Config.NUM_CLASSES)

    def forward(self, x):
        return self.backbone(x)
