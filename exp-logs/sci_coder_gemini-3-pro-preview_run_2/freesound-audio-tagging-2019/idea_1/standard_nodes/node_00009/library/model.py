import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from library.config import Config


class AudioMobileNet(nn.Module):
    """
    AudioMobileNet using Transfer Learning from ImageNet.
    Uses MobileNetV2 with V2 weights (Cite solution_lesson_node_00008).
    Adapts first layer for 1-channel input by summing weights (Cite solution_lesson_node_00006).
    """

    def __init__(self):
        super(AudioMobileNet, self).__init__()

        # Load pre-trained model with modern V2 recipe
        weights = MobileNet_V2_Weights.IMAGENET1K_V2
        self.base_model = mobilenet_v2(weights=weights)

        # Modify the first convolutional layer to accept 1 channel instead of 3
        # We sum the weights along the channel dimension to preserve filter structure
        old_conv = self.base_model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # Initialize new conv weights by summing original RGB weights
        with torch.no_grad():
            new_conv.weight.data = old_conv.weight.data.sum(dim=1, keepdim=True)

        self.base_model.features[0][0] = new_conv

        # Replace the classifier head
        # MobileNetV2 classifier is Sequential(Dropout, Linear)
        # We keep the dropout and replace the Linear layer
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
        # Using Max Pooling instead of default Average Pooling to avoid signal dilution
        # from zero-padding in variable length audio (Cite solution_lesson_node_00002)
        x = nn.functional.adaptive_max_pool2d(x, (1, 1))

        # Flatten
        x = torch.flatten(x, 1)

        # Classification
        x = self.base_model.classifier(x)

        return x
