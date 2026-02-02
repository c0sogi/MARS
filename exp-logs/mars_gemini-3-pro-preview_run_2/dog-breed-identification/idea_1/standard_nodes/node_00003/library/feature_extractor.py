import torch
import torch.nn as nn
from torchvision import models
from library.utils import set_seed
from library.config import SEED


class DogResNet(nn.Module):
    """
    ResNet50 model modified for Dog Breed Classification.
    """

    def __init__(self, num_classes):
        super().__init__()
        set_seed(SEED)

        # Use ImageNet V2 weights if available for better initial features
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        except AttributeError:
            weights = models.ResNet50_Weights.IMAGENET1K_V1

        self.backbone = models.resnet50(weights=weights)

        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
