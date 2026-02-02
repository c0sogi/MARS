import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class AudioResNet34(nn.Module):
    """
    ResNet34-based model for Audio Tagging.
    Uses a pretrained ResNet34 backbone adapted for single-channel spectrograms.
    Cite {solution_lesson_node_00004}
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, in_channels=1):
        super(AudioResNet34, self).__init__()

        # Load pretrained ResNet34
        # Using weights parameter for modern torchvision
        try:
            from torchvision.models import ResNet34_Weights

            weights = ResNet34_Weights.IMAGENET1K_V1
            self.model = models.resnet34(weights=weights)
        except ImportError:
            self.model = models.resnet34(pretrained=True)

        # Modify first layer for 1-channel input (Spectrogram)
        # Original: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modify last layer for number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
