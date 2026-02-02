import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class WhaleResNet(nn.Module):
    """
    ResNet-18 based model for Right Whale call detection.
    Modified to accept single-channel spectrograms and output binary logits.
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        """
        Initialize the WhaleResNet model.

        Args:
            pretrained (bool): If True, load weights pretrained on ImageNet.
        """
        super(WhaleResNet, self).__init__()

        # Load the ResNet-18 architecture
        # Using the modern weights API if available in the environment
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.model = models.resnet18(weights=weights)

        # 1. Adapt the first convolutional layer for single-channel input
        # Standard ResNet-18 conv1 is Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        # If using pretrained weights, initialize the new 1-channel kernel
        # by averaging the weights of the original 3-channel kernel.
        # This preserves the learned filters (edge detectors, etc.) better than random initialization.
        if pretrained and weights is not None:
            with torch.no_grad():
                self.model.conv1.weight.data = original_conv1.weight.data.mean(
                    dim=1, keepdim=True
                )

        # 2. Modify the final fully connected layer
        # Replace the 1000-class output with NUM_CLASSES (1 for binary classification)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)

        Returns:
            torch.Tensor: Logits of shape (batch_size, 1).
                          Apply sigmoid externally for probabilities.
        """
        return self.model(x)
