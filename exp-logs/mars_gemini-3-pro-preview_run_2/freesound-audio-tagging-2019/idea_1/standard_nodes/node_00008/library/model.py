import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from library.config import Config


class AudioMobileNet(nn.Module):
    """
    MobileNetV2 based model using Transfer Learning from ImageNet.
    Adapted for 1-channel Audio Spectrograms using Kernel Summation.
    """

    def __init__(self):
        super(AudioMobileNet, self).__init__()

        # Load pre-trained MobileNetV2
        # Cite solution_lesson_node_00006
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        self.base_model = mobilenet_v2(weights=weights)

        # Modify the first layer to accept 1 channel instead of 3
        # We sum the weights of the RGB channels to preserve the filters' spatial structure
        # Cite solution_lesson_node_00006
        original_conv = self.base_model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        with torch.no_grad():
            new_conv.weight[:] = original_conv.weight.sum(dim=1, keepdim=True)
            if original_conv.bias is not None:
                new_conv.bias[:] = original_conv.bias

        self.base_model.features[0][0] = new_conv

        # Modify the classifier head for the specific number of audio classes
        # MobileNetV2 classifier is a Sequential(Dropout, Linear)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrogram of shape (Batch, 1, Freq, Time)
        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes)
        """
        # Feature Extraction
        x = self.base_model.features(x)

        # Global Max Pooling
        # We replace the standard Global Average Pooling to avoid signal dilution
        # caused by zero-padding in variable length audio clips.
        # Cite solution_lesson_node_00002
        x = nn.functional.adaptive_max_pool2d(x, (1, 1))

        # Flatten
        x = torch.flatten(x, 1)

        # Classification
        x = self.base_model.classifier(x)

        return x
