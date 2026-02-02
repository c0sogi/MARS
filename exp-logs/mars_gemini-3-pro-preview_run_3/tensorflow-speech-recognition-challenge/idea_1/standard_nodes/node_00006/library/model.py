import torch
import torch.nn as nn
import timm
from library import config


class SimpleAudioCNN(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        """
        Wrapper for a ResNet-18 model using timm, adapted for 1-channel audio input.
        Replaces the previous custom CNN to improve feature extraction capabilities.
        """
        super(SimpleAudioCNN, self).__init__()

        # Use ResNet34 with pretrained weights.
        # in_chans=1 adapts the first layer to accept single-channel spectrograms.
        # Cite solution_lesson_node_00005: Preserving pretrained weights via adaptation.
        self.model = timm.create_model(
            "resnet34", pretrained=True, in_chans=1, num_classes=num_classes
        )

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 1, n_mels, time_steps).
        """
        return self.model(x)
