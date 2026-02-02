import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class AudioMobileNet(nn.Module):
    """
    Lightweight 2D CNN based on MobileNetV2 architecture.
    Adapted for Audio Spectrogram inputs using Pre-trained Weights.
    """

    def __init__(self):
        super(AudioMobileNet, self).__init__()

        # Load pre-trained MobileNetV2
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Adapt first convolutional layer for 1-channel input
        # Original: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        original_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,  # 1
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # Initialize the new conv layer with the sum of the original weights
        # This preserves the magnitude of activations for grayscale/1-channel inputs derived from RGB
        with torch.no_grad():
            self.model.features[0][0].weight.data = original_conv.weight.data.sum(
                dim=1, keepdim=True
            )

        # Modify Classifier
        # MobileNetV2 classifier: Sequential(Dropout(p=0.2), Linear(in_features=1280, out_features=1000))
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrogram of shape (Batch, 1, Freq, Time)
        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes)
        """
        # Feature Extraction
        x = self.model.features(x)

        # Global Max Pooling
        # Collapses spatial dimensions (Freq, Time) to (1, 1)
        # Uses Max Pooling to avoid signal dilution from zero-padding (Cite solution_lesson_node_00002)
        x = nn.functional.adaptive_max_pool2d(x, (1, 1))

        # Flatten
        x = torch.flatten(x, 1)

        # Classification
        x = self.model.classifier(x)

        return x
