import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class ResNet34(nn.Module):
    """
    ResNet34 backbone for Audio Tagging.
    Uses ImageNet pretrained weights and modifies the input/output layers
    for spectrograms and multi-label classification.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, in_channels=1):
        """
        Initialize the ResNet34 model.

        Args:
            num_classes (int): Number of output classes (default: 80).
            in_channels (int): Number of input channels (default: 1 for mono spectrograms).
        """
        super(ResNet34, self).__init__()

        # Load pretrained ResNet34 (Cite solution_lesson_node_00004)
        weights = models.ResNet34_Weights.IMAGENET1K_V1
        self.base = models.resnet34(weights=weights)

        # Modify first conv layer to accept 1 channel instead of 3
        # We sum the weights of the original 3 channels to preserve energy/magnitude
        original_conv1 = self.base.conv1
        self.base.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            self.base.conv1.weight.data = original_conv1.weight.data.sum(
                dim=1, keepdim=True
            )

        # Replace Global Average Pooling with Global Max Pooling (Cite solution_lesson_node_00005)
        # This is crucial for weakly supervised detection of sparse events.
        self.base.avgpool = nn.AdaptiveMaxPool2d((1, 1))

        # Modify Classification Head
        in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input spectrogram of shape (Batch, Channels, Freq, Time).

        Returns:
            torch.Tensor: Output logits of shape (Batch, Num_Classes).
        """
        return self.base(x)
