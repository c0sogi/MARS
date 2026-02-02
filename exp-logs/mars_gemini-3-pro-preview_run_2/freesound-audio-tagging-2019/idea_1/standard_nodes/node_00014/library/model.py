import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from library.config import Config


class AudioMobileNet(nn.Module):
    """
    MobileNetV2 with Transfer Learning and custom adaptation for Audio.
    """

    def __init__(self):
        super(AudioMobileNet, self).__init__()

        # Cite Lesson 8: Modern Pre-training Recipes (V2 Weights)
        weights = MobileNet_V2_Weights.DEFAULT
        self.model = mobilenet_v2(weights=weights)

        # Cite Lesson 6: Cross-Modal Transfer Learning via Kernel Summation
        # Modify the first layer to accept 1 channel instead of 3
        first_conv = self.model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,  # 1
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )

        # Initialize new kernel by summing weights along the channel dimension
        with torch.no_grad():
            new_conv.weight[:] = first_conv.weight.sum(dim=1, keepdim=True)
            if first_conv.bias is not None:
                new_conv.bias[:] = first_conv.bias

        self.model.features[0][0] = new_conv

        # Replace the classifier head
        # MobileNetV2 classifier is (Dropout, Linear)
        self.model.classifier[1] = nn.Linear(
            self.model.last_channel, Config.NUM_CLASSES
        )

    def forward(self, x):
        # Feature Extraction
        x = self.model.features(x)

        # Cite Lesson 2: Signal Dilution in GAP
        # Use Adaptive Max Pooling instead of default Global Average Pooling
        x = nn.functional.adaptive_max_pool2d(x, (1, 1))

        # Flatten
        x = torch.flatten(x, 1)

        # Classification
        x = self.model.classifier(x)

        return x
