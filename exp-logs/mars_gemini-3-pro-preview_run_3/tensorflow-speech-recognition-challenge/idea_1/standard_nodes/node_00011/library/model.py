import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from library import config


class SimpleAudioCNN(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        """
        Wrapper for a ResNet-34 model using torchvision, manually adapted for 1-channel audio input.
        Cite solution_lesson_node_00010: Explicitly sums weights of the first conv layer.
        """
        super(SimpleAudioCNN, self).__init__()

        # Use ResNet34 with pretrained weights
        weights = ResNet34_Weights.DEFAULT
        self.model = resnet34(weights=weights)

        # Modify first layer for 1-channel input
        # Original: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_conv1 = self.model.conv1

        # Create new conv1 with 1 input channel
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        # Cite solution_lesson_node_00010: Sum weights across input channels to preserve feature detection
        with torch.no_grad():
            self.model.conv1.weight.data = original_conv1.weight.data.sum(
                dim=1, keepdim=True
            )

        # Modify fc layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 1, n_mels, time_steps).
        """
        return self.model(x)
