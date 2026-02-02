import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from library.config import Config


class AudioMobileNet(nn.Module):
    """
    Audio classification model based on MobileNetV3-Small.
    Adapts the pre-trained ImageNet model for multi-label audio tagging.
    """

    def __init__(self, pretrained=True):
        """
        Args:
            pretrained (bool): If True, loads weights pre-trained on ImageNet.
        """
        super(AudioMobileNet, self).__init__()

        # Determine weights configuration
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        else:
            weights = None

        # Load the backbone model
        self.model = mobilenet_v3_small(weights=weights)

        # The input data is expected to be [Batch, 3, Freq, Time].
        # The dataset pipeline handles repeating the single-channel spectrogram
        # to 3 channels, so the first layer (Conv2d expecting 3 channels) remains unchanged.

        # Modify the classifier head
        # The original classifier is a Sequential block typically structured as:
        # (0): Linear(576 -> 1024)
        # (1): Hardswish
        # (2): Dropout
        # (3): Linear(1024 -> 1000)

        # We replace the final Linear layer to output the correct number of classes (80).
        # We access the classifier Sequential container.
        classifier = self.model.classifier

        # Identify the last layer (the readout layer)
        last_layer_index = len(classifier) - 1
        last_layer = classifier[last_layer_index]

        # Ensure the last layer is indeed a Linear layer before replacing
        if isinstance(last_layer, nn.Linear):
            in_features = last_layer.in_features
            # Replace with new Linear layer for our specific number of classes
            classifier[last_layer_index] = nn.Linear(in_features, Config.NUM_CLASSES)
        else:
            # Fallback mechanism if architecture differs slightly, though unlikely for standard mobilenet_v3
            # We assume the last item is the classification layer.
            # If strictly following standard implementation, hardcoding 1024 is also an option,
            # but reading in_features is safer.
            raise ValueError(
                "Expected last layer of MobileNetV3 classifier to be nn.Linear"
            )

        # Re-assign the modified classifier to the model
        self.model.classifier = classifier

        # Replace Average Pooling with Max Pooling to handle weakly labeled data
        # Cite solution_lesson_node_00002
        self.model.avgpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input spectrograms of shape (Batch, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch, NUM_CLASSES).
        """
        return self.model(x)
